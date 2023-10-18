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
            break;
        };
        let Ok(dist_sample) = dist_sink.pull_sample() else {
            break;
        };
        
        let ref_buffer = ref_sample.buffer().unwrap();
        let dist_buffer = dist_sample.buffer().unwrap();

        let pts_diff = dist_buffer.pts().unwrap() - ref_buffer.pts().unwrap();
        if let Some(previous_pts_diff) = previous_pts_diff {
            assert_eq!(pts_diff, previous_pts_diff);
        }
        previous_pts_diff = Some(pts_diff);

        let mut ref_pic: VmafPicture = unsafe { MaybeUninit::zeroed().assume_init() };
        alloc_init_vmaf_pic(&mut ref_pic, &ref_buffer);

        let mut dist_pic: VmafPicture = unsafe { MaybeUninit::zeroed().assume_init() };
        alloc_init_vmaf_pic(&mut dist_pic, &dist_buffer);

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

fn alloc_init_vmaf_pic(pic: &mut VmafPicture, buffer: &gst::BufferRef) {
    let meta = buffer.meta::<gst_video::VideoMeta>().unwrap();
    let format = meta.format();
    let format_info = gst_video::VideoFormatInfo::from(format);

    let width = meta.width();
    let height = meta.height();
    let depthes = format_info.depth();
    let depth = depthes[0];

    let (vmaf_pixel_format, format_depth) = match format {
        gst_video::VideoFormat::Gray8 => (VmafPixelFormat::VMAF_PIX_FMT_YUV400P, 8),
        gst_video::VideoFormat::I420 => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 8),
        gst_video::VideoFormat::Y42b => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 8),
        gst_video::VideoFormat::Y444 => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 8),
        gst_video::VideoFormat::I42010le => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 10),
        gst_video::VideoFormat::I42210le => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 10),
        gst_video::VideoFormat::Y44410le => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 10),
        gst_video::VideoFormat::I42012le => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 12),
        gst_video::VideoFormat::I42212le => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 12),
        gst_video::VideoFormat::Y44412le => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 12),
        gst_video::VideoFormat::Gray16Le => (VmafPixelFormat::VMAF_PIX_FMT_YUV400P, 16),
        _ => panic!("Pixel format not supported: {}", format),
    };
    assert_eq!(format_depth, depth);

    unsafe {
        let r = vmaf_picture_alloc(pic as *mut VmafPicture, vmaf_pixel_format, depth, width, height);
        assert_eq!(r, 0);
    };

    // See Also https://github.com/GStreamer/gst-plugins-base/blob/master/gst-libs/gst/video/video-format.c

    let offsets = meta.offset();
    let strides = meta.stride();

    let n_planes = format_info.n_planes();
    let n_components = format_info.n_components();
    let planes = format_info.plane();
    let pixel_strides = format_info.pixel_stride();
    let n_value_bytes = depth.div_ceil(8) as usize;

    assert_eq!(depth, pic.bpc);

    if 1 < n_value_bytes {
        assert!(format_info.is_le(), "Multi bytes pixel format must be little endian: {}", format);
    }
    for shift in format_info.shift() {
        assert_eq!(*shift, 0, "The formats have bit shift not supported: {}", format);
    }

    assert_eq!(format_info.bits(), depth, "In my understanding, in little endian format, bits must be the same as depth: {}", format);
    assert_eq!(format_info.n_planes(), n_components, "Video format must be planar format: {}", format);
    assert_eq!(planes.len(), n_planes as usize);
    assert_eq!(n_planes, n_components);

    if format_info.is_yuv() {
        assert_eq!(n_planes, 3, "Yuv video format must have 3 planar components: {}", format);
        assert_eq!((planes[0], planes[1], planes[2]), (0, 1, 2), "must be Y U V order: {}", format);
        assert_eq!(depthes[0], depthes[1], "Yuv must have all the same bit depth: {}", format);
        assert_eq!(depthes[1], depthes[2], "Yuv must have all the same bit depth: {}", format);
    } else if format_info.is_gray() {
        assert_eq!(n_planes, 1, "Gray video format must be single component: {}", format);
        assert_eq!(planes[0], 0);
    } else {
        panic!("Video format must be gray or yuv: {}", format);
    }

    let map = buffer.map_readable().unwrap();
    let src_all_data = map.as_slice();

    for component_index in 0..n_components {
        let component_index = component_index as usize;
        let component_width = format_info.scale_width(component_index as u8, width) as usize;
        let component_height = format_info.scale_width(component_index as u8, height) as usize;

        assert_eq!(pic.w[component_index] as usize, component_width);
        assert_eq!(pic.h[component_index] as usize, component_height);

        let src_offset = offsets[component_index] as usize;
        let src_pixel_stride = pixel_strides[component_index] as usize;
        let src_stride = strides[component_index] as usize;
        assert_eq!(src_pixel_stride, n_value_bytes, "Pixel stride must be depth.div_ceil(8): {}", format);
        assert!(component_width < src_stride);

        let dst_stride = pic.stride[component_index] as usize;
        assert!(component_width < dst_stride);

        let src_data = src_all_data[src_offset] as *const u8;
        let dst_data = pic.data[component_index] as *mut u8;

        for y in 0..component_height {
            let n_copy_bytes = n_value_bytes * component_width as usize;
            let src_offset = y * (src_stride * n_value_bytes) as usize;
            let dst_offset = y * (dst_stride * n_value_bytes) as usize;
            unsafe {
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

