using UnityEngine;


[DefaultExecutionOrder(100)]
[RequireComponent(typeof(Rigidbody))]
public class ShipController : MonoBehaviour
{
    [Header("Motion (set from config)")]
    public float speed          = 0f;   // target speed
    public float headingDegrees = 0f;   // target yaw

    [Header("Ground truth")]
    public int classId = 8;

    [Header("Speed PI gains")]
    public float Kp = 1000f;
    [Tooltip("Integral gain. Accumulates speed-error over time to cancel steady-state water drag.")]
    public float Ki = 200f;
    [Tooltip("Clamp on the integrator to prevent windup if the ship is blocked.")]
    public float integralClamp = 15000f;

    [Header("Heading control")]
    [Tooltip("Torque gain for the heading-correction autopilot (same role as AutoForwardEngine.steeringCorrectionForce).")]
    public float steeringCorrectionForce = 50f;

    Rigidbody _rb;
    float     _integral;

    public void Init(Vector3 startPos, float headingDeg, float speedMps, int cls)
    {
        headingDegrees = headingDeg;
        speed          = speedMps;
        classId        = cls;

        transform.position = startPos;
        transform.rotation = Quaternion.Euler(0f, headingDegrees, 0f);
    }

    void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        if (_rb != null)
        {
            _rb.isKinematic = false;
            _rb.useGravity  = true;
        }

        // AutoForwardEngine is superseded by this component
        // it is disable to avoid two conflicting thrust sources.
        var engine = GetComponent<AutoForwardEngine>();
        if (engine != null) engine.enabled = false;
    }

    void FixedUpdate()
    {
        // Speed PI controller 
        float currentSpeed = Vector3.Dot(_rb.velocity, transform.forward);
        float error        = speed - currentSpeed;
        _integral = Mathf.Clamp(_integral + error * Time.fixedDeltaTime,
                                -integralClamp, integralClamp);

        float thrust = Kp * error + Ki * _integral;
        _rb.AddForce(transform.forward * thrust, ForceMode.Force);

        // Heading autopilot 
        float headingError = Mathf.DeltaAngle(transform.eulerAngles.y, headingDegrees);
        _rb.AddTorque(Vector3.up * headingError * steeringCorrectionForce * Time.fixedDeltaTime,
                      ForceMode.VelocityChange);
    }
}
