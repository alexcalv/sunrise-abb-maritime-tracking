using System;
using System.IO;
using UnityEditor.Recorder;
using UnityEditor.Recorder.Encoder;
using UnityEditor.Recorder.Input;
using UnityEngine;

public class MultiCameraRecorder
{
    RecorderController _controller;

    //Build one MP4 recorder per camera, bound to that camera's RenderTexture
    public MultiCameraRecorder(SimulationDirector director, bool verbose)
    {
        RecorderOptions.VerboseMode = verbose;
        _controller = Build(director);
    }

    public void Start()
    {
        _controller.PrepareRecording();
        if (!_controller.StartRecording())
            throw new Exception("RecorderController.StartRecording() returned false (see console for details).");
    }

    public void Stop()
    {
        _controller?.StopRecording();
    }

    static RecorderController Build(SimulationDirector director)
    {
        var cfg = director.Config;

        var settings = ScriptableObject.CreateInstance<RecorderControllerSettings>();
        settings.SetRecordModeToManual();
        settings.FrameRatePlayback = FrameRatePlayback.Constant;
        settings.FrameRate         = cfg.video.fps;
        settings.CapFrameRate      = true;
        settings.ExitPlayMode      = false;

        foreach (var rig in director.Cameras)
        {
            var movie = ScriptableObject.CreateInstance<MovieRecorderSettings>();
            movie.name    = rig.name;
            movie.Enabled = true;
            movie.EncoderSettings = new CoreEncoderSettings
            {
                Codec           = CoreEncoderSettings.OutputCodec.MP4,
                EncodingQuality = CoreEncoderSettings.VideoEncodingQuality.High,
            };
            movie.CaptureAudio = false;
            movie.ImageInputSettings = new RenderTextureInputSettings
            {
                RenderTexture   = rig.renderTexture,
                FlipFinalOutput = false,
            };
            movie.OutputFile = Path.Combine(director.OutDir, rig.name).Replace('\\', '/');

            settings.AddRecorderSettings(movie);
        }

        return new RecorderController(settings);
    }
}
