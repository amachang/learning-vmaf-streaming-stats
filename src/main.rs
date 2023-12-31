use std::{fmt, error::Error, path::{Path, PathBuf}, ffi::CString, ptr, mem::MaybeUninit, intrinsics::copy_nonoverlapping};
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use gst::prelude::*;
use glib;
use clap::Parser;
use lazy_static::lazy_static;
use stats::OnlineStats;
use libvmaf_sys::*;
use log;
use env_logger;

const VMAF_SCORE_INTERVAL: u32 = 100;

lazy_static! {
    static ref ELEMENT_FACTORY_FILESRC: gst::ElementFactory = gst::ElementFactory::find("filesrc").expect("filesrc must be installed");
    static ref ELEMENT_FACTORY_DECODEBIN: gst::ElementFactory = gst::ElementFactory::find("decodebin").expect("decodebin must be installed");
    static ref ELEMENT_FACTORY_DEINTERLACE: gst::ElementFactory = gst::ElementFactory::find("deinterlace").expect("deinterlace must be installed");
    static ref ELEMENT_FACTORY_TEE: gst::ElementFactory = gst::ElementFactory::find("tee").expect("tee must be installed");
    static ref ELEMENT_FACTORY_QUEUE: gst::ElementFactory = gst::ElementFactory::find("queue").expect("queue must be installed");
    static ref ELEMENT_FACTORY_APPSINK: gst::ElementFactory = gst::ElementFactory::find("appsink").expect("appsink must be installed");
    static ref ELEMENT_FACTORY_X265ENC: gst::ElementFactory = gst::ElementFactory::find("x265enc").expect("x265enc must be installed");
    static ref ELEMENT_FACTORY_H265PARSE: gst::ElementFactory = gst::ElementFactory::find("h265parse").expect("h265parse must be installed");
    static ref ELEMENT_FACTORY_AVDEC_H265: gst::ElementFactory = gst::ElementFactory::find("avdec_h265").expect("avdec_h265 must be installed");

    static ref QUARK_STRUCTURE_NAME_VIDEO: glib::Quark = glib::Quark::from_str("video/x-raw");
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg()]
    input_video_path: PathBuf,

    #[arg(long, default_value="28.0")]
    crf: Crf,

    #[arg(long, default_value_t=10)]
    start_seconds: u64,
}

#[derive(Debug, Clone, Copy)]
struct Crf {
    deci_crf: u16,
}

impl Crf {
    fn round_to_crf(f: f64) -> Self {
        Self { deci_crf: (f * 10f64).round() as u16 }
    }
}

impl Into<f64> for &Crf {
    fn into(self: Self) -> f64 {
        self.deci_crf as f64 / 10f64
    }
}

impl fmt::Display for Crf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let float: f64 = self.into();
        write!(f, "{}", float)
    }
}

impl From<&str> for Crf {
    fn from(s: &str) -> Self {
        // TODO too rough impl
        Self::round_to_crf(s.parse::<f64>().unwrap())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    let cli = Cli::parse();

    let mut vmaf_conf: VmafConfiguration = unsafe { MaybeUninit::zeroed().assume_init() };
    vmaf_conf.log_level = VmafLogLevel::VMAF_LOG_LEVEL_DEBUG;
    vmaf_conf.n_threads = 16;

    let mut vmaf_model_conf: VmafModelConfig = unsafe { MaybeUninit::zeroed().assume_init() };
    let mut vmaf_model: *mut VmafModel = ptr::null_mut();
    let model_version_cstr = CString::new("vmaf_v0.6.1").unwrap();
    unsafe {
        let r = vmaf_model_load(&mut vmaf_model as *mut *mut VmafModel, &mut vmaf_model_conf as *mut VmafModelConfig, model_version_cstr.as_ptr());
        assert_eq!(r, 0);
    }

    let mut vmaf_ctx: *mut VmafContext = ptr::null_mut();

    gst::init()?;

    log::trace!("Start preparation for pipeline");
    let pipeline = prepare_pipeline(cli.input_video_path, cli.crf)?;
    log::trace!("Pipeline prepared");

    pipeline.seek(1.0, gst::SeekFlags::FLUSH, gst::SeekType::Set, gst::ClockTime::from_seconds(cli.start_seconds), gst::SeekType::End, gst::ClockTime::from_seconds(0))?;

    pipeline.set_state(gst::State::Playing)?;

    let ref_sink = pipeline.by_name("ref_sink").unwrap().dynamic_cast::<gst_app::AppSink>().unwrap();
    let dist_sink = pipeline.by_name("dist_sink").unwrap().dynamic_cast::<gst_app::AppSink>().unwrap();

    unsafe {
        let r = vmaf_init(&mut vmaf_ctx as *mut *mut VmafContext, vmaf_conf);
        assert_eq!(r, 0);
        let r = vmaf_use_features_from_model(vmaf_ctx, vmaf_model);
        assert_eq!(r, 0);
    }

    let mut stats = OnlineStats::new();

    let mut sample_count = 0;
    let mut previous_pts_diff = None;
    while !ref_sink.is_eos() && !dist_sink.is_eos() {
        let Ok(ref_sample) = ref_sink.pull_sample() else {
            continue;
        };
        let Ok(dist_sample) = dist_sink.pull_sample() else {
            continue;
        };
       
        let ref_buffer = ref_sample.buffer().unwrap();
        let dist_buffer = dist_sample.buffer().unwrap();

        let pts_diff = dist_buffer.pts().unwrap() - ref_buffer.pts().unwrap();
        if let Some(previous_pts_diff) = previous_pts_diff {
            assert_eq!(pts_diff, previous_pts_diff);
        }
        previous_pts_diff = Some(pts_diff);

        let mut ref_pic: VmafPicture = unsafe { MaybeUninit::zeroed().assume_init() };
        alloc_init_vmaf_pic(&mut ref_pic, &ref_sample);

        let mut dist_pic: VmafPicture = unsafe { MaybeUninit::zeroed().assume_init() };
        alloc_init_vmaf_pic(&mut dist_pic, &dist_sample);

        unsafe {
            log::trace!("Read pictures");
            let r = vmaf_read_pictures(vmaf_ctx, &mut ref_pic as *mut VmafPicture, &mut dist_pic as *mut VmafPicture, sample_count);
            assert_eq!(r, 0);

            // check both pictures freed
            assert_eq!(ref_pic.ref_, ptr::null_mut());
            assert_eq!(dist_pic.ref_, ptr::null_mut());
        }
        sample_count += 1;

        if sample_count >= VMAF_SCORE_INTERVAL {
            log::trace!("Flush scores");
            unsafe {
                // flushing
                let r = vmaf_read_pictures(vmaf_ctx, ptr::null_mut(), ptr::null_mut(), 0);
                assert_eq!(r, 0);

                /*
                for sample_index in 0..sample_count {
                    let mut score: f64 = 0.0f64;
                    let r = vmaf_score_at_index(vmaf_ctx, vmaf_model, &mut score as *mut f64, sample_index);
                    assert_eq!(r, 0);
                    stats.add(score);
                }
                */

                let mut score: f64 = 0.0f64;
                let r = vmaf_score_pooled(vmaf_ctx, vmaf_model, VmafPoolingMethod::VMAF_POOL_METHOD_HARMONIC_MEAN, &mut score as *mut f64, 0, sample_count - 1);
                assert_eq!(r, 0);
                stats.add(score);

                if 1 < stats.len() {
                    let stddev = stats.stddev();
                    let stderr = stddev / (stats.len() as f64).sqrt();
                    // 1.96: 95% conf interval
                    // 1.64: 90% conf interval
                    let conf_interval = stderr * 1.64;
                    log::info!("Vmaf: {} ± {} (samples={}, stddev={}, stderr={})", stats.mean(), conf_interval, stats.len(), stddev, stderr);
                    if conf_interval < 0.5 {
                        break;
                    }
                }

                let r = vmaf_close(vmaf_ctx);
                assert_eq!(r, 0);
                let r = vmaf_init(&mut vmaf_ctx as *mut *mut VmafContext, vmaf_conf);
                assert_eq!(r, 0);
                let r = vmaf_use_features_from_model(vmaf_ctx, vmaf_model);
                assert_eq!(r, 0);
            }

            sample_count = 0;
        }
    };

    unsafe {
        let r = vmaf_close(vmaf_ctx);
        assert_eq!(r, 0);
    }

    pipeline.set_state(gst::State::Null)?;


    Ok(())
}

fn alloc_init_vmaf_pic(pic: &mut VmafPicture, sample: &gst::Sample) {
    // See Also https://github.com/GStreamer/gst-plugins-base/blob/master/gst-libs/gst/video/video-format.c

    let buffer = sample.buffer().unwrap();
    let caps = sample.caps().unwrap();
    let info = gst_video::VideoInfo::from_caps(caps).unwrap();
    let format = info.format();
    let format_info = gst_video::VideoFormatInfo::from(format);

    let width = info.width() as usize;
    let height = info.height() as usize;
    let depth = info.comp_depth(0);

    let n_components = info.n_components();
    let pixel_stride = depth.div_ceil(8) as usize;

    if 1 < pixel_stride {
        assert!(format_info.is_le(), "Multi bytes pixel format must be little endian: {}", format);
    }
    for shift in format_info.shift() {
        assert_eq!(*shift, 0, "The formats have bit shift not supported: {}", format);
    }

    assert_eq!(format_info.bits(), depth, "In my understanding, in little endian format, bits must be the same as depth: {}", format);
    assert_eq!(format_info.n_planes(), n_components, "Video format must be planar format: {}", format);

    if format_info.is_yuv() {
        assert_eq!(n_components, 3, "Yuv video format must have 3 planar components: {}", format);
        assert_eq!(info.comp_depth(0), info.comp_depth(1), "Yuv must have all the same bit depth: {}", format);
        assert_eq!(info.comp_depth(1), info.comp_depth(2), "Yuv must have all the same bit depth: {}", format);
    } else if format_info.is_gray() {
        assert_eq!(n_components, 1, "Gray video format must be single component: {}", format);
    } else {
        panic!("Video format must be gray or yuv: {}", format);
    }

    let map = buffer.map_readable().unwrap();
    let src_all_data = map.as_slice();

    let mut src_component_infos = Vec::new();

    for component_index in 0..n_components {
        let width = info.comp_width(component_index as u8) as usize;
        let height = info.comp_height(component_index as u8) as usize;
        let pixel_stride = info.comp_pstride(component_index as u8) as usize;
        let offset = info.comp_offset(component_index as u8) as usize;
        let stride = info.comp_stride(component_index as u8) as usize;
        src_component_infos.push(ComponentInfo { width, height, pixel_stride, offset, stride });
    }

    let (vmaf_pixel_format, format_depth, format_n_components) = match format {
        gst_video::VideoFormat::Gray8 => (VmafPixelFormat::VMAF_PIX_FMT_YUV400P, 8, 1),
        gst_video::VideoFormat::I420 => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 8, 3),
        gst_video::VideoFormat::Y42b => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 8, 3),
        gst_video::VideoFormat::Y444 => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 8, 3),
        gst_video::VideoFormat::I42010le => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 10, 3),
        gst_video::VideoFormat::I42210le => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 10, 3),
        gst_video::VideoFormat::Y44410le => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 10, 3),
        gst_video::VideoFormat::I42012le => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 12, 3),
        gst_video::VideoFormat::I42212le => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 12, 3),
        gst_video::VideoFormat::Y44412le => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 12, 3),
        gst_video::VideoFormat::Gray16Le => (VmafPixelFormat::VMAF_PIX_FMT_YUV400P, 16, 1),
        _ => panic!("Pixel format not supported: {}", format),
    };
    assert_eq!(format_depth, depth);
    assert_eq!(format_n_components, n_components);

    let src_video_info = VideoInfo { vmaf_pixel_format, width, height, depth };

    alloc_init_vmaf_pic_impl(pic, src_all_data, src_video_info, src_component_infos);
}

struct VideoInfo {
    vmaf_pixel_format: VmafPixelFormat,
    width: usize,
    height: usize,
    depth: u32,
}

struct ComponentInfo {
    width: usize,
    height: usize,
    pixel_stride: usize,
    offset: usize,
    stride: usize
}

fn alloc_init_vmaf_pic_impl(pic: &mut VmafPicture, src_all_data: &[u8], src_video_info: VideoInfo, src_component_infos: Vec<ComponentInfo>) {
    unsafe {
        let r = vmaf_picture_alloc(
            pic as *mut VmafPicture,
            src_video_info.vmaf_pixel_format,
            src_video_info.depth,
            src_video_info.width as u32,
            src_video_info.height as u32,
            );
        assert_eq!(r, 0);
    };

    let pixel_stride = pic.bpc.div_ceil(8) as usize;

    for (component_index, src_component_info) in src_component_infos.into_iter().enumerate() {
        assert_eq!(src_component_info.pixel_stride, pixel_stride);

        let component_index = component_index as usize;
        let component_width = src_component_info.width;
        let component_height = src_component_info.height;

        assert_eq!(pic.w[component_index] as usize, component_width);
        assert_eq!(pic.h[component_index] as usize, component_height);

        let src_stride = src_component_info.stride;
        assert!(component_width <= src_stride);

        let dst_stride = pic.stride[component_index] as usize;
        assert!(component_width <= dst_stride);

        let src_data = &src_all_data[src_component_info.offset] as *const u8;
        let dst_data = pic.data[component_index] as *mut u8;

        for y in 0..component_height {
            let n_copy_bytes =  component_width * pixel_stride;
            let src_offset = y * src_stride * pixel_stride;
            let dst_offset = y * dst_stride * pixel_stride;
            unsafe {
                /*
                log::trace!("Start copy src={:?} dst={:?} bytes={:}",
                    src_data.wrapping_add(src_offset),
                    dst_data.wrapping_add(dst_offset),
                    n_copy_bytes);
                */
                copy_nonoverlapping(
                    src_data.wrapping_add(src_offset),
                    dst_data.wrapping_add(dst_offset),
                    n_copy_bytes);
            };
        }
    }
}


fn prepare_pipeline(path: impl AsRef<Path>, crf: Crf) -> Result<gst::Pipeline, Box<dyn Error>> {
    let path = path.as_ref();
    let src_el = ELEMENT_FACTORY_FILESRC.create().name("src").property("location", path).build()?;
    let decoder_el = ELEMENT_FACTORY_DECODEBIN.create().name("decoder").property("force-sw-decoders", true).build()?;
    let deinterlace_el = ELEMENT_FACTORY_DEINTERLACE.create().name("deinterlace").build()?;
    let tee_el = ELEMENT_FACTORY_TEE.create().name("tee").build()?;

    let ref_queue_el = ELEMENT_FACTORY_QUEUE.create().name("ref_queue")
        .property("max-size-buffers", u32::MAX)
        .property("max-size-bytes", 1_000_000_000u32)
        .property("max-size-time", u64::MAX)
        .build()?;
    let ref_sink_el = ELEMENT_FACTORY_APPSINK.create().name("ref_sink").property("sync", false).build()?;

    let encoder_queue_el = ELEMENT_FACTORY_QUEUE.create().name("encoder_queue")
        .property("max-size-buffers", u32::MAX)
        .property("max-size-bytes", 1_000_000_000u32)
        .property("max-size-time", u64::MAX)
        .build()?;

    let encoder_el = ELEMENT_FACTORY_X265ENC.create().name("encoder").property("option-string", &format!("crf={}", crf)).build()?;
    let encoder_parser_el = ELEMENT_FACTORY_H265PARSE.create().name("encoder_parser").build()?;
    let dist_decoder_el = ELEMENT_FACTORY_AVDEC_H265.create().name("dist_decoder").build()?;
    let dist_queue_el = ELEMENT_FACTORY_QUEUE.create().name("dist_queue")
        .property("max-size-buffers", u32::MAX)
        .property("max-size-bytes", 1_000_000_000u32)
        .property("max-size-time", u64::MAX)
        .build()?;
    let dist_sink_el = ELEMENT_FACTORY_APPSINK.create().name("dist_sink").property("sync", false).build()?;

    let pipeline = gst::Pipeline::builder().name("main_pipeline").build();

    pipeline.add_many(&[
        &src_el, &decoder_el, &deinterlace_el, &tee_el,
        &ref_queue_el, &ref_sink_el,
        &encoder_queue_el, &encoder_el, &encoder_parser_el, &dist_decoder_el, &dist_queue_el, &dist_sink_el,
    ])?;

    gst::Element::link_many(&[&src_el, &decoder_el])?;
    gst::Element::link_many(&[&deinterlace_el, &tee_el])?;
    gst::Element::link_many(&[&tee_el, &ref_queue_el, &ref_sink_el])?;
    gst::Element::link_many(&[&tee_el, &encoder_queue_el, &encoder_el, &encoder_parser_el, &dist_decoder_el, &dist_queue_el, &dist_sink_el])?;

    decoder_el.connect_pad_added(move |_, src_pad| {
        let Some(caps) = src_pad.current_caps() else {
            return;
        };
        if !caps.iter().any(|s| s.name_quark() == *QUARK_STRUCTURE_NAME_VIDEO) {
            return;
        };

        let decoder_target_sink_pad = deinterlace_el.static_pad("sink").unwrap();
        src_pad.link(&decoder_target_sink_pad).unwrap();
    });

    let bus = pipeline.bus().unwrap();
    pipeline.set_state(gst::State::Paused)?;
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        match msg.view() {
            gst::MessageView::StateChanged(state_changed) => {
                if msg.src().map_or(false, |src| src.is::<gst::Pipeline>()) {
                    if state_changed.pending() == gst::State::VoidPending && state_changed.current() >= gst::State::Paused {
                        break;
                    }
                }
            },
            gst::MessageView::Eos(_) => {
                break;
            },
            gst::MessageView::Error(err) => {
                panic!("Error message arrived: {:?} {:?} {:?}",
                    msg.src().map(|src| src.clone()),
                    err.error(),
                    err.debug().map(|s| s.to_string()));
            },
            _ => (),
        }
    }

    let decoder_target_sink_pad = pipeline.by_name("deinterlace").unwrap().static_pad("sink").unwrap();
    assert!(decoder_target_sink_pad.is_linked());

    Ok(pipeline)
}

