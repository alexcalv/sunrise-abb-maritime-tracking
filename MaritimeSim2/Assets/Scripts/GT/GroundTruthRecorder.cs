using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;

// Records per-frame ground truth (MOT .txt + JSON) for every "Vessel"-tagged
// object seen by renderCamera: image-space bbox, occlusion, stable track id.
public class GroundTruthRecorder : MonoBehaviour
{
    [Header("Scene References")]
    public Camera renderCamera;

    [Header("Output Settings")]
    public string outputFolder   = "Assets/GT_Output/";
    public string sequenceName   = "sequence_01";
    public bool   writeMOT       = true;
    public bool   writeJSON      = true;

    [Header("Capture")]
    [Tooltip("When true, capture one frame automatically every LateUpdate (hand-built scenes). " +
             "The generator pipeline sets this false and calls CaptureFrame() itself for exact frame alignment.")]
    public bool autoCapture = true;

    [Header("Occlusion Sampling")]
    [Range(8, 64)]
    public int   occlusionRaySamples = 16;
    [Tooltip("A vessel is flagged 'occluded' when its occlusion ratio exceeds this.")]
    [Range(0f, 1f)]
    public float occludedThreshold = 0.5f;

    [Header("Filtering")]
    public float minBBoxWidth  = 5f;   // px — skip detections smaller than this
    public float minBBoxHeight = 5f;

    [Header("Classification")]
    [Tooltip("MOT 'class' column used for a vessel that has no ShipController.")]
    public int defaultClassId = 8;

    [Header("Bounds Shrink (0=no change, 0.3=30% smaller)")]
    [Tooltip("Fallback only — used when a vessel has no TightBounds. Vessels with TightBounds ignore this.")]
    public Vector3 boundsShrink = Vector3.zero;

    private List<FrameRecord>    _allFrames  = new List<FrameRecord>();
    private StreamWriter         _motWriter;
    private int                  _frameIndex = 1;   // MOT is 1-based
    private bool                 _closed;

    private Dictionary<int, int> _trackIdMap  = new Dictionary<int, int>();
    private int                  _nextTrackId = 1;

    private const int OcclusionSeed = 12345;
    private System.Random _rng;


    void Start()
    {
        if (renderCamera == null)
            renderCamera = GetComponent<Camera>();

        _rng = new System.Random(OcclusionSeed);

        Directory.CreateDirectory(outputFolder);

        if (writeMOT)
        {
            string motPath = Path.Combine(outputFolder, sequenceName + ".txt");
            _motWriter = new StreamWriter(motPath, false, new System.Text.UTF8Encoding(false));
            Debug.Log($"[GT] MOT file → {motPath}");
        }
    }

    void LateUpdate()
    {
        if (autoCapture)
            CaptureFrame(_frameIndex++);
    }


    // Capture one frame of annotations for every "Vessel"-tagged object.
    // Called automatically when autoCapture is true, or driven explicitly by
    // the SimulationDirector so video and ground-truth frames stay aligned.
    public void CaptureFrame(int frameId)
    {
        var frame = new FrameRecord
        {
            frame_id    = frameId,
            timestamp   = Time.time,
            annotations = new List<VesselAnnotation>()
        };

        foreach (var vessel in GameObject.FindGameObjectsWithTag("Vessel"))
        {
            var ann = BuildAnnotation(vessel, frameId);
            if (ann == null) continue;

            frame.annotations.Add(ann);

            if (writeMOT && _motWriter != null)
                _motWriter.WriteLine(ToMOTLine(ann));
        }

        if (writeJSON)
            _allFrames.Add(frame);
    }

    VesselAnnotation BuildAnnotation(GameObject vessel, int frameId)
    {
        Bounds wb;
        var tight = vessel.GetComponent<TightBounds>();
        if (tight != null)
            wb = tight.GetWorldBounds();
        else
        {
            // fallback to renderer bounds
            Renderer[] renderers = vessel.GetComponentsInChildren<Renderer>();
            if (renderers.Length == 0) return null;

            // 1. World AABB
            wb = renderers[0].bounds;
            foreach (var r in renderers) wb.Encapsulate(r.bounds);

            wb = new Bounds(wb.center, Vector3.Scale(wb.size,
                new Vector3(1f - boundsShrink.x,
                            1f - boundsShrink.y,
                            1f - boundsShrink.z)));
        }


        Rect bbox = WorldBoundsToImageRect(wb, renderCamera);
        if (bbox == Rect.zero)              return null;
        if (bbox.width  < minBBoxWidth)     return null;
        if (bbox.height < minBBoxHeight)    return null;

        float occRatio  = EstimateOcclusion(wb, vessel);
        float visibility = Mathf.Clamp01(1f - occRatio);

        int instanceId = vessel.GetInstanceID();
        if (!_trackIdMap.TryGetValue(instanceId, out int trackId))
        {
            trackId = _nextTrackId++;
            _trackIdMap[instanceId] = trackId;
        }

        var ship = vessel.GetComponent<ShipController>();
        int classId = ship != null ? ship.classId : defaultClassId;

        return new VesselAnnotation
        {
            frame_id        = frameId,
            track_id        = trackId,
            label           = vessel.name,
            bbox_x          = bbox.x,
            bbox_y          = bbox.y,
            bbox_w          = bbox.width,
            bbox_h          = bbox.height,
            confidence      = 1.0f,
            class_id        = classId,
            occlusion_ratio = occRatio,
            visibility      = visibility,
            occluded        = occRatio > occludedThreshold,
        };
    }

 
    // Projects world AABB corners to screen space and returns a Rect
    // in IMAGE coordinates (origin top-left, Y increases downward).
    // Returns Rect.zero if the object is behind the camera.
    Rect WorldBoundsToImageRect(Bounds b, Camera cam)
    {
        Vector3 c = b.center, e = b.extents;
        Vector3[] corners =
        {
            cam.WorldToScreenPoint(c + new Vector3( e.x,  e.y,  e.z)),
            cam.WorldToScreenPoint(c + new Vector3(-e.x,  e.y,  e.z)),
            cam.WorldToScreenPoint(c + new Vector3( e.x, -e.y,  e.z)),
            cam.WorldToScreenPoint(c + new Vector3(-e.x, -e.y,  e.z)),
            cam.WorldToScreenPoint(c + new Vector3( e.x,  e.y, -e.z)),
            cam.WorldToScreenPoint(c + new Vector3(-e.x,  e.y, -e.z)),
            cam.WorldToScreenPoint(c + new Vector3( e.x, -e.y, -e.z)),
            cam.WorldToScreenPoint(c + new Vector3(-e.x, -e.y, -e.z)),
        };

        float minX =  float.MaxValue, minY =  float.MaxValue;
        float maxX = -float.MaxValue, maxY = -float.MaxValue;

        foreach (var p in corners)
        {
            if (p.z < 0f) return Rect.zero;   // any corner behind cam → skip
            minX = Mathf.Min(minX, p.x); maxX = Mathf.Max(maxX, p.x);
            minY = Mathf.Min(minY, p.y); maxY = Mathf.Max(maxY, p.y);
        }

        int imgHeight = cam.targetTexture != null ? cam.targetTexture.height : cam.pixelHeight;

        float imgY = imgHeight - maxY;

        return new Rect(minX, imgY, maxX - minX, maxY - minY);
    }


    // Estimates occlusion by sampling points on the vessel AABB and
    // raycasting from the camera. Returns [0..1] fraction of blocked rays.
    float EstimateOcclusion(Bounds wb, GameObject self)
    {
        Vector3 camPos = renderCamera.transform.position;
        int blocked = 0;

        for (int i = 0; i < occlusionRaySamples; i++)
        {
            Vector3 sample = new Vector3(
                RandomRange(wb.min.x, wb.max.x),
                RandomRange(wb.min.y, wb.max.y),
                RandomRange(wb.min.z, wb.max.z));

            Vector3 dir  = sample - camPos;
            float   dist = dir.magnitude - 0.05f;   // small epsilon

            if (Physics.Raycast(camPos, dir.normalized, out RaycastHit hit, dist))
                if (!hit.collider.transform.IsChildOf(self.transform))
                    blocked++;
        }

        return (float)blocked / occlusionRaySamples;
    }

    // Uniform float in [min, max) from the seeded RNG.
    float RandomRange(float min, float max) => min + (float)_rng.NextDouble() * (max - min);


    // Formats one annotation as a MOT line:
    // frame, track_id, x, y, w, h, confidence, class_id, visibility
    string ToMOTLine(VesselAnnotation a)
    {
        return string.Format(
            "{0},{1},{2:F4},{3:F4},{4:F4},{5:F4},{6:F6},{7},{8}",
            a.frame_id,
            a.track_id,
            a.bbox_x,
            a.bbox_y,
            a.bbox_w,
            a.bbox_h,
            a.confidence,
            a.class_id,
            1.0 - a.occlusion_ratio
        );
    }

    void OnApplicationQuit()
    {
        FinalizeAndClose();
    }

    public void FinalizeAndClose()
    {
        if (_closed) return;
        _closed = true;

        if (_motWriter != null)
        {
            _motWriter.Flush();
            _motWriter.Close();
            _motWriter = null;
        }

        if (writeJSON && _allFrames.Count > 0)
        {
            string jsonPath = Path.Combine(outputFolder, sequenceName + ".json");
            var sb = new StringBuilder();
            sb.AppendLine("[");
            for (int fi = 0; fi < _allFrames.Count; fi++)
            {
                var f = _allFrames[fi];
                sb.AppendLine("  {");
                sb.AppendLine($"    \"frame_id\": {f.frame_id},");
                sb.AppendLine($"    \"timestamp\": {f.timestamp:F4},");
                sb.AppendLine("    \"annotations\": [");
                for (int ai = 0; ai < f.annotations.Count; ai++)
                {
                    var a = f.annotations[ai];
                    string comma = (ai < f.annotations.Count - 1) ? "," : "";
                    sb.AppendLine("      {");
                    sb.AppendLine($"        \"track_id\": {a.track_id},");
                    sb.AppendLine($"        \"label\": \"{a.label}\",");
                    sb.AppendLine($"        \"bbox_x\": {a.bbox_x:F4},");
                    sb.AppendLine($"        \"bbox_y\": {a.bbox_y:F4},");
                    sb.AppendLine($"        \"bbox_w\": {a.bbox_w:F4},");
                    sb.AppendLine($"        \"bbox_h\": {a.bbox_h:F4},");
                    sb.AppendLine($"        \"confidence\": {a.confidence:F6},");
                    sb.AppendLine($"        \"class_id\": {a.class_id},");
                    sb.AppendLine($"        \"occlusion_ratio\": {a.occlusion_ratio:F4},");
                    sb.AppendLine($"        \"visibility\": {a.visibility:F4},");
                    sb.AppendLine($"        \"occluded\": {a.occluded.ToString().ToLower()}");
                    sb.AppendLine("      }" + comma);
                }
                sb.AppendLine("    ]");
                string frameComma = (fi < _allFrames.Count - 1) ? "  }," : "  }";
                sb.AppendLine(frameComma);
            }
            sb.AppendLine("]");
            File.WriteAllText(jsonPath, sb.ToString());
            Debug.Log($"[GT] JSON file → {jsonPath}  ({_allFrames.Count} frames)");
        }

        Debug.Log("[GT] Recording complete.");
    }
}

// ── Data structures ──────────────────────────────────────────

[System.Serializable]
public class FrameRecord
{
    public int                  frame_id;
    public float                timestamp;
    public List<VesselAnnotation> annotations;
}

[System.Serializable]
public class VesselAnnotation
{
    public int    frame_id;
    public int    track_id;
    public string label;
    public float  bbox_x, bbox_y, bbox_w, bbox_h;
    public float  confidence;
    public int    class_id;
    public float  occlusion_ratio;
    public float  visibility;
    public bool   occluded;
}