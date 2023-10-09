use std::{error::Error, path::{Path, PathBuf}, ffi::CString, ptr, mem::MaybeUninit, intrinsics::copy_nonoverlapping};
use gstreamer as gst;
use gstreamer_app as gst_app;
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
    let pipeline = prepare_pipeline(cli.input_video_path)?;

    pipeline.seek(1.0, gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT, gst::SeekType::Set, gst::ClockTime::from_seconds(0), gst::SeekType::Set, gst::ClockTime::from_seconds(60))?;

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

        let (ref_pts, ref_width, ref_height, ref_vmaf_pixel_format, ref_pixel_depth) = metadata_for_sample(&ref_sample);
        let (dist_pts, dist_width, dist_height, dist_vmaf_pixel_format, dist_pixel_depth) = metadata_for_sample(&dist_sample);

        let pts_diff = dist_pts - ref_pts;
        if let Some(previous_pts_diff) = previous_pts_diff {
            assert_eq!(pts_diff, previous_pts_diff);
        };
        previous_pts_diff = Some(pts_diff);

        assert_eq!(ref_width, dist_width);
        assert_eq!(ref_height, dist_height);
        assert_eq!(ref_vmaf_pixel_format, dist_vmaf_pixel_format);
        assert_eq!(ref_pixel_depth, dist_pixel_depth);

        let mut ref_pic: VmafPicture = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut dist_pic: VmafPicture = unsafe { MaybeUninit::zeroed().assume_init() };
        unsafe {
            let r = vmaf_picture_alloc(&mut ref_pic as *mut VmafPicture, ref_vmaf_pixel_format, ref_pixel_depth as u32, ref_width as u32, ref_height as u32);
            assert_eq!(r, 0);
            let r = vmaf_picture_alloc(&mut dist_pic as *mut VmafPicture, dist_vmaf_pixel_format, dist_pixel_depth as u32, dist_width as u32, dist_height as u32);
            assert_eq!(r, 0);
        };

        copy_sample_data_to_vmaf_pic(&ref_sample, &ref_pic, ref_width, ref_height, ref_vmaf_pixel_format, ref_pixel_depth);
        copy_sample_data_to_vmaf_pic(&dist_sample, &dist_pic, dist_width, dist_height, dist_vmaf_pixel_format, dist_pixel_depth);

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
            // flushing
            unsafe {
                let r = vmaf_read_pictures(vmaf_ctx, ptr::null_mut(), ptr::null_mut(), 0);
                assert_eq!(r, 0);

                for sample_index in 0..sample_count {
                    let mut score: f64 = 0.0f64;
                    let r = vmaf_score_at_index(vmaf_ctx, vmaf_model, &mut score as *mut f64, sample_index);
                    stats.add(score);
                    assert_eq!(r, 0);
                }

                let stddev = stats.stddev();
                let stderr = stddev / (stats.len() as f64).sqrt();
                log::info!("Vmaf: {} ± {} (stddev={}, stderr={})", stats.mean(), stderr * 1.96, stddev, stderr);

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

fn copy_sample_data_to_vmaf_pic(sample: &gst::Sample, pic: &VmafPicture, width: usize, height: usize, vmaf_pixel_format: VmafPixelFormat, pixel_depth: u8) {
    let resolution = width * height;

    let buffer = sample.buffer().unwrap();
    let map = buffer.map_readable().unwrap();
    let data = map.as_slice();

    let (y_pixel_count, u_pixel_count, v_pixel_count) = match vmaf_pixel_format {
        VmafPixelFormat::VMAF_PIX_FMT_YUV400P => {
            (resolution, 0, 0)
        },
        VmafPixelFormat::VMAF_PIX_FMT_YUV420P => {
            assert_eq!(width % 2, 0);
            assert_eq!(height % 2, 0);
            assert_eq!(resolution % 4, 0);
            (resolution, resolution / 4, resolution / 4)
        },
        VmafPixelFormat::VMAF_PIX_FMT_YUV422P => {
            assert_eq!(width % 2, 0);
            assert_eq!(resolution % 2, 0);
            (resolution, resolution / 2, resolution / 2)
        },
        VmafPixelFormat::VMAF_PIX_FMT_YUV444P => {
            (resolution, resolution, resolution)
        },
        VmafPixelFormat::VMAF_PIX_FMT_UNKNOWN => panic!("Pixel format unknown"),
    };

    let y_bit_len = y_pixel_count * pixel_depth as usize;
    let u_bit_len = u_pixel_count * pixel_depth as usize;
    let v_bit_len = v_pixel_count * pixel_depth as usize;

    let y_data_len = y_bit_len / 8 + if y_bit_len % 8 == 0 { 0 } else { 1 };
    let u_data_len = u_bit_len / 8 + if u_bit_len % 8 == 0 { 0 } else { 1 };
    let v_data_len = v_bit_len / 8 + if v_bit_len % 8 == 0 { 0 } else { 1 };

    assert_eq!(data.len(), (y_data_len + u_data_len + v_data_len) as usize);

    let y_data = &data[0 .. y_data_len];
    let u_data = &data[y_data_len .. y_data_len+u_data_len];
    let v_data = &data[y_data_len+u_data_len .. y_data_len+u_data_len+v_data_len];

    unsafe {
        copy_nonoverlapping(y_data.as_ptr() as *const _, pic.data[0], y_data.len());
        copy_nonoverlapping(u_data.as_ptr() as *const _, pic.data[1], u_data.len());
        copy_nonoverlapping(v_data.as_ptr() as *const _, pic.data[2], v_data.len());
    }
}

fn prepare_pipeline(path: impl AsRef<Path>) -> Result<gst::Pipeline, Box<dyn Error>> {
    let path = path.as_ref();
    let src_el = ELEMENT_FACTORY_FILESRC.create().name("src").property("location", path).build()?;
    let decoder_el = ELEMENT_FACTORY_DECODEBIN.create().name("decoder").property("force-sw-decoders", true).build()?;
    let deinterlace_el = ELEMENT_FACTORY_DEINTERLACE.create().name("deinterlace").build()?;
    let tee_el = ELEMENT_FACTORY_TEE.create().name("tee").build()?;

    let ref_queue_el = ELEMENT_FACTORY_QUEUE.create().name("ref_queue")
        .property("max-size-buffers", u32::MAX)
        .property("max-size-bytes", 100_000_000u32)
        .property("max-size-time", u64::MAX)
        .build()?;
    let ref_sink_el = ELEMENT_FACTORY_APPSINK.create().name("ref_sink").property("sync", false).build()?;

    let encoder_queue_el = ELEMENT_FACTORY_QUEUE.create().name("encoder_queue")
        .property("max-size-buffers", u32::MAX)
        .property("max-size-bytes", 100_000_000u32)
        .property("max-size-time", u64::MAX)
        .build()?;
    let encoder_el = ELEMENT_FACTORY_X265ENC.create().name("encoder").property("option-string", "crf=51").build()?;
    let encoder_parser_el = ELEMENT_FACTORY_H265PARSE.create().name("encoder_parser").build()?;
    let dist_decoder_el = ELEMENT_FACTORY_AVDEC_H265.create().name("dist_decoder").build()?;
    let dist_queue_el = ELEMENT_FACTORY_QUEUE.create().name("dist_queue")
        .property("max-size-buffers", u32::MAX)
        .property("max-size-bytes", 100_000_000u32)
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

fn metadata_for_sample(sample: &gst::Sample) -> (gst::ClockTime, usize, usize, VmafPixelFormat, u8) {
    let buffer = sample.buffer().unwrap();
    let pts = buffer.pts().unwrap();

    let caps = sample.caps().unwrap();
    for structure in caps.iter() {
        if structure.name_quark() != *QUARK_STRUCTURE_NAME_VIDEO {
            continue;
        };

        let format = structure.get::<&str>("format").unwrap();
        let width = structure.get::<i32>("width").unwrap() as usize;
        let height = structure.get::<i32>("height").unwrap() as usize;

        let (vmaf_pixel_format, pixel_depth) = vmaf_pixel_format_info(format);

        return (pts, width, height, vmaf_pixel_format, pixel_depth);
    }
    panic!("Pixel format not supported");
}

fn vmaf_pixel_format_info(format: &str) -> (VmafPixelFormat, u8) {
    match format {
        "I420" => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 8),
        "NV12" => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 8),
        "YV12" => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 8),
        "Y42B" => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 8),
        "Y444" => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 8),
        "I420_10LE" => (VmafPixelFormat::VMAF_PIX_FMT_YUV420P, 10),
        "I422_10LE" => (VmafPixelFormat::VMAF_PIX_FMT_YUV422P, 10),
        "Y444_10LE" => (VmafPixelFormat::VMAF_PIX_FMT_YUV444P, 10),
        _ => panic!("Pixel format not supported"),
    }
}

