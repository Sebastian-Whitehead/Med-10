using System.Collections;
using System.Collections.Generic;
using UnityEditor.PackageManager.UI;
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

public class ClientSpawner : MonoBehaviour
{

    [Header("Spawn Settings")]
    public List<GameObject> spawnList = new List<GameObject>();
    public int min_spawn_count = 1;
    public int max_spawn_count = 5;
    public Vector3 spawnRange = new Vector3(10, 0, 10);

    [Header("Speed Settings")]
    private float totalSpeed;

    private List<GameObject> spawnedObjects = new List<GameObject>();
    private int respawnFrameCount = 0;
    private bool respawn = false;
    private bool hasMoved = false;
    private bool hasCaptured = true;


    // Update is called once per frame
    void Update()
    {

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
    /// Checks the speed of spawned objects and triggers capture if conditions are met.
    /// </summary>
    public float CheckSpeedOfSpawnedObjects()
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

        return totalSpeed;
    }

    /// <summary>
    /// Draws the spawn area in the Unity Editor for visualization.
    /// </summary>
    private void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireCube(transform.position, new Vector3(spawnRange.x * 2, spawnRange.y * 2, spawnRange.z * 2));
    }

}