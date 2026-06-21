using UnityEngine;

// Legacy propulsion for hand-built scenes. Superseded by ShipController, which
// disables this component on ships spawned by the SimulationDirector.
[RequireComponent(typeof(Rigidbody))]
public class AutoForwardEngine : MonoBehaviour
{
    [Header("Engine Settings")]
    public float thrustPower = 2000f;
    
    [Header("Auto-Pilot Settings")]
    public float steeringCorrectionForce = 50f;
    private float targetHeading;
    private Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        targetHeading = transform.eulerAngles.y; 
    }

    void FixedUpdate()
    {
        rb.AddForce(transform.forward * thrustPower, ForceMode.Force);

        float currentHeading = transform.eulerAngles.y;
        float headingError = Mathf.DeltaAngle(currentHeading, targetHeading);

        rb.AddTorque(Vector3.up * headingError * steeringCorrectionForce * Time.fixedDeltaTime,
                     ForceMode.VelocityChange);
    }
}
