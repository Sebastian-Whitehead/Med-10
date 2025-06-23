using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TpPoint : MonoBehaviour
{
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(transform.position, 0.2f); // Draw a sphere with a radius of 0.2
    }
}
