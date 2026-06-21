using UnityEngine;

public class TightBounds : MonoBehaviour
{
    [Header("Box Definition (local space)")]
    public Vector3 center = Vector3.zero;
    public Vector3 size   = Vector3.one;

    [Header("Gizmo Display")]
    public Color gizmoColor         = new Color(0f, 1f, 0f, 0.35f);   // semi-transparent green fill
    public Color gizmoWireColor     = new Color(0f, 1f, 0f, 1f);      // solid green wire
    public bool  showOnlyWhenSelected = true;

    // Public API used by GroundTruthRecorder
    public Bounds GetWorldBounds()
    {
        Vector3 worldCenter = transform.TransformPoint(center);
        Vector3 worldSize   = Vector3.Scale(size, AbsScale(transform.lossyScale));
        return new Bounds(worldCenter, worldSize);
    }

    // Gizmos 
    void OnDrawGizmosSelected()
    {
        DrawGizmo();
    }

    void OnDrawGizmos()
    {
        if (!showOnlyWhenSelected)
            DrawGizmo();
    }

    void DrawGizmo()
    {
        Matrix4x4 prev = Gizmos.matrix;
        Gizmos.matrix = Matrix4x4.TRS(
            transform.TransformPoint(center),
            transform.rotation,
            AbsScale(transform.lossyScale));

        Gizmos.color = gizmoColor;
        Gizmos.DrawCube(Vector3.zero, size);

        Gizmos.color = gizmoWireColor;
        Gizmos.DrawWireCube(Vector3.zero, size);

        Gizmos.matrix = prev;
    }

    static Vector3 AbsScale(Vector3 s) =>
        new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
}