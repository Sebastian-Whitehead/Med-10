using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RestCamRandom : MonoBehaviour
{
    public GameObject Cam;
    public GameObject Lookie;
    public List<GameObject> positions = new List<GameObject>();
    public Vector3 l = new Vector3(10, 10, 10);
    public float y_randomization = 0.05f;



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
        // Draw a sphere at each of the positions in the list
        foreach (GameObject position in positions)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawSphere(position.transform.position, 0.05f);
        }

        Gizmos.color = Color.magenta;
        Gizmos.DrawWireCube(Lookie.transform.position, new Vector3(l.x * 2, l.y * 2, l.z * 2));
    }

    void CameraPosition()
    {
        // get a random gameobject from the list
        int randomIndex = Random.Range(0, positions.Count);
        GameObject randomPosition = positions[randomIndex];
        // set the camera position to the random gameobject with a slight y offset
        float y_rand = Random.Range(-y_randomization, y_randomization);
        Cam.transform.position = new Vector3(randomPosition.transform.position.x, randomPosition.transform.position.y + y_rand, randomPosition.transform.position.z);
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
