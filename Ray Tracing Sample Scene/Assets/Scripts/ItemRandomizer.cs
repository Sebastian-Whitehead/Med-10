using System.Collections;
using System.Collections.Generic;
using UnityEditor.PackageManager.UI;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class ItemRandomizer : MonoBehaviour
{

    [Header("Randomizer Settings")]
    public RandomTeleport randomTeleport;
    public ModelChanger tableChanger;

    [Header("Spawn Settings")]
    public List<GameObject> spawnList = new List<GameObject>();
    public int min_spawn_count = 1;
    public int max_spawn_count = 5;
    public Vector3 spawnRange = new Vector3(10, 0, 10);
    

    [Header("Capture Settings")]
    public PerceptionCamera perceptionCamera;

    public Volume PTvolume;
    [Tooltip("Ensure to adjust the corresponding variable in the PathTracing camera as well.")] public int sample = 512;
    private bool PT_Enabled = false;
    public int captureCount = 0;
    public int captureLimit = 100;

    [Header("Speed Settings")]
    public float speedThreshold = 0.1f;
    private float totalSpeed;

    private List<GameObject> spawnedObjects = new List<GameObject>();
    private int respawnFrameCount = 0;
    private bool respawn = false;
    private bool hasMoved = false;
    private bool hasCaptured = true;

    public void Start()
    {
        PTvolume.profile.TryGet(out PathTracing pathTracingVolume);

        PT_Enabled = sample != -1;
        
        pathTracingVolume.maximumSamples.Override(sample);
        PTvolume.gameObject.SetActive(PT_Enabled );
        perceptionCamera.useAccumulation = PT_Enabled;
        print($"Path Tracing Enabled: {PT_Enabled}");
    }

    // Update is called once per frame
    void Update()
    {
        CheckCaptureCount();

        if (Input.GetKeyDown(KeyCode.Space))
        {
            HandleRandomization();
        }

        CheckSpeedOfSpawnedObjects();
        HandleRespawn();
    }

    /// <summary>
    /// Checks if the capture count has reached the limit and exits if necessary.
    /// </summary>
    private void CheckCaptureCount()
    {
        if (captureCount == -1) return;

        if (captureCount > captureLimit)
        {
            Debug.Log("Capture limit reached. Exiting...");

#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false; // Stops play mode in the Editor
#else
            Application.Quit(); // Quits the application in a standalone build
#endif
        }
    }

    /// <summary>
    /// Spawns a random number of objects within the defined spawn range.
    /// </summary>
    public void SpawnRandomObjects()
    {
        DestroySpawnedObjects();
        
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

            GameObject spawnedObject = Instantiate(prefab, spawnPosition, prefab.transform.rotation);
            spawnedObjects.Add(spawnedObject);
        }
    }

    /// <summary>
    /// Destroys all currently spawned objects.
    /// </summary>
    public void DestroySpawnedObjects()
    {
        foreach (GameObject obj in spawnedObjects)
        {
            Destroy(obj);
        }
        spawnedObjects.Clear();
    }

    /// <summary>
    /// Handles the respawn logic with a delay.
    /// </summary>
    private void HandleRespawn()
    {
        if (!respawn) return;
        print($"Rendering frame {captureCount}");
        // Wait until the path tracing frame is complete

        respawnFrameCount++;
        if (PT_Enabled && respawnFrameCount > sample + 1)
        {
            HandleRandomization();
            respawnFrameCount = 0;
            respawn = false;
        }else if (!PT_Enabled && respawnFrameCount > 10)
        {
            HandleRandomization();
            respawnFrameCount = 0;
            respawn = false;
        }
    }

    public int GetAccumulationSamples()
{
    // Access the active volume stack
    var volumeStack = VolumeManager.instance.stack;

    // Retrieve the PathTracing component from the volume stack
    PathTracing pathTracing = volumeStack.GetComponent<PathTracing>();
    if (pathTracing != null && pathTracing.active)
    {
        return pathTracing.maximumSamples.value; // Access the maximum samples value
    }

    Debug.LogWarning("PathTracing component not found or not active in the volume stack.");
    return 0; // Default value if not found
}

    /// <summary>
    /// Checks the speed of spawned objects and triggers capture if conditions are met.
    /// </summary>
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
        
        perceptionCamera.RequestCapture(); // Request a capture from the PerceptionCamera
        captureCount++;
        
        respawn = true;
    }
}

    /// <summary>
    /// Draws the spawn area in the Unity Editor for visualization.
    /// </summary>
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(transform.position, new Vector3(spawnRange.x * 2, spawnRange.y * 2, spawnRange.z * 2));
    }

    private void HandleRandomization()
    {
        randomTeleport?.Teleport();
        perceptionCamera.GetComponent<CameraRandomizer>()?.MoveCameraRandomly();
        tableChanger?.RandomModel();
        SpawnRandomObjects();
    }
}