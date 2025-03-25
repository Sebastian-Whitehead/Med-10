using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRandomizer : MonoBehaviour
{
    public GameObject Cam;
    public GameObject Lookie;
    BoxCollider box_l;
    public float c_x;
    public float c_y;
    public float c_z;
    Vector3 l_pos;



    // Start is called before the first frame update
    void Start()
    {
        CameraPosition();
        Vector3 point = ObjectPosition();
        CameraLookAt(point);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(transform.position, new Vector3(c_x * 2, c_y * 2, c_z * 2));
    }

    void CameraPosition()
    {
        var position_c = new Vector3(Random.Range(-c_x, c_x), Random.Range(-c_y, c_y), Random.Range(-c_z, c_z));
        Cam.transform.position += position_c;
        Debug.Log(position_c);
    }

    Vector3 ObjectPosition()
    {
        box_l = Lookie.GetComponent<BoxCollider>();

        Vector3 center = box_l.bounds.center;
        Vector3 extents = box_l.bounds.extents;

        // Generate random offsets within the bounds
        float randomX = Random.Range(-extents.x, extents.x);
        float randomY = Random.Range(-extents.y, extents.y);
        float randomZ = Random.Range(-extents.z, extents.z);

        // Compute the final position
        Vector3 randomPoint = center + new Vector3(randomX, randomY, randomZ);
        Debug.Log(randomPoint);
        return randomPoint;
    }

    void CameraLookAt(Vector3 point)
    {
        Cam.transform.LookAt(point);
    }
}

