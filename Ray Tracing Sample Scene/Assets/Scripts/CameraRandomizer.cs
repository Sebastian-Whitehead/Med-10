using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRandomizer : MonoBehaviour
{
    public GameObject Cam;
    public GameObject Lookie;
    public GameObject Zone;
    
    public Vector3 c = new Vector3(10, 10, 10);
    public Vector3 l = new Vector3(10, 10, 10);
    Vector3 l_pos;



    // Start is called before the first frame update
    void Start()
    {
        MoveCameraRandomly();
    }

    public void MoveCameraRandomly()
    {
        CameraPosition();
        Vector3 point = ObjectPosition();
        CameraLookAt(point);
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(Zone.transform.position, new Vector3( c.x* 2, c.y * 2, c.z * 2));
        

        Gizmos.color = Color.red;
        Gizmos.DrawWireCube(Lookie.transform.position, new Vector3(l.x * 2, l.y * 2, l.z * 2));
    }

    void CameraPosition()
    {
        var position_c = new Vector3(Random.Range(-c.x, c.x), Random.Range(-c.y, c.y), Random.Range(-c.z, c.z));
        Cam.transform.position = Zone.transform.position + position_c;
    }

    Vector3 ObjectPosition()
    {
        

        Vector3 center = Lookie.transform.position;
        Debug.Log(center);

        // Generate random offsets within the bounds
        float randomX = Random.Range(-l.x, l.x);
        float randomY = Random.Range(-l.y, l.y);
        float randomZ = Random.Range(-l.z, l.z);

        // Compute the final position
        Vector3 randomPoint = center + new Vector3(randomX, randomY, randomZ);
        
        return randomPoint;
    }

    void CameraLookAt(Vector3 point)
    {
        Cam.transform.LookAt(point);
    }
}

