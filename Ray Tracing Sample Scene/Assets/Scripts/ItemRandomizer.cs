using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

public class ItemRandomizer : MonoBehaviour
{
    public List<GameObject> spawnList = new List<GameObject>();
    public int min_spawn_count, max_spawn_count;
    private List<GameObject> spawnedObjects = new List<GameObject>();
    public Vector3 spawnRange = new Vector3(10, 0, 10);
    public PerceptionCamera perceptionCamera;
    public int captureCount = 0;
    public int captureLimit = 100;

    public bool triggerCapture = false;


    public float speedThreshold = 0.1f;
    public float totalSpeed;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        checkCaptureCount();
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SpawnRandomObjects();
            perceptionCamera.GetComponent<CameraRandomizer>().MoveCameraRandomly();
        }
        CheckSpeedOfSpawnedObjects();
        respawnObjects();

    }

    private void checkCaptureCount()
    {
        if (captureCount == -1) return;
        else if (captureCount >= captureLimit)
        {
            Debug.Log("Capture limit reached. Exiting...");

        #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false; // Stops play mode in the Editor
        #else
            Application.Quit(); // Quits the application in a standalone build
        #endif

        }
    }
    public void SpawnRandomObjects()
    {
        DestroySpawnedObjects();
        checkCaptureCount();
        int spawnCount = Random.Range(min_spawn_count, max_spawn_count + 1);
        for (int i = 0; i < spawnCount; i++)
        {
            int randomIndex = Random.Range(0, spawnList.Count);
            GameObject prefab = spawnList[randomIndex];
            Vector3 spawnPosition = new Vector3(
                transform.position.x + Random.Range(-spawnRange.x, spawnRange.x),
                transform.position.y + Random.Range(-spawnRange.y, spawnRange.y),
                transform.position.z + Random.Range(-spawnRange.z, spawnRange.z)
            );
            GameObject spawnedObject = Instantiate(prefab);
            spawnedObject.transform.position = spawnPosition;
            spawnedObjects.Add(spawnedObject);
        }
    }

    public void DestroySpawnedObjects()
    {
        foreach (GameObject obj in spawnedObjects)
        {
            Destroy(obj);
        }
        spawnedObjects.Clear();
    }

    // Draw the spawn area in the editor
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(transform.position, new Vector3(spawnRange.x * 2, spawnRange.y * 2, spawnRange.z * 2));
    }


    // Delay the respawn of objects to ensure the capture is complete before spawning new objects
    private int respawnFrameCount = 0;
    bool respawn = false;
    private void respawnObjects()
    {
        if (!respawn) return;

        respawnFrameCount++;
        if (respawnFrameCount > 100)
        {
            SpawnRandomObjects();
            respawnFrameCount = 0;
            respawn = false;
        }
    }
    
    bool hasMoved = false;
    bool hasCaptured = true;
    private void CheckSpeedOfSpawnedObjects()
    {
        totalSpeed = 0;
        foreach (GameObject obj in spawnedObjects)
        {    
            Rigidbody rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                totalSpeed += rb.velocity.magnitude;
            }
        }

        if (totalSpeed > speedThreshold)
        {
            hasMoved = true;
            hasCaptured = false;
        }
        else if (totalSpeed < speedThreshold && hasMoved && !hasCaptured)
        {
            hasCaptured = true;
            hasMoved = false;

            if (triggerCapture)
            {
                perceptionCamera.RequestCapture();
                captureCount++;
            }
            respawn = true;
        }
    }
}