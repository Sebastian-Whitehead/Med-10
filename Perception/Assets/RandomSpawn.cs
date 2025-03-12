using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomSpawn : MonoBehaviour
{
    public List<GameObject> spawnList = new List<GameObject>();
    public int min_spawn_count, max_spawn_count;
    private List<GameObject> spawnedObjects = new List<GameObject>();
    public Vector3 spawnRange = new Vector3(10, 0, 10);


    public float speedThreshold = 0.1f;
    public float totalSpeed;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            SpawnRandomObjects();
        }
        CheckSpeedOfSpawnedObjects();
    }

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
            Quaternion spawnRotation = Random.rotation;
            GameObject spawnedObject = Instantiate(prefab, spawnPosition, spawnRotation);
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

    private void CheckSpeedOfSpawnedObjects()
    {
        foreach (GameObject obj in spawnedObjects)
        {
            Rigidbody rb = obj.GetComponent<Rigidbody>();
            if (rb != null)
            {
                totalSpeed += rb.velocity.magnitude;
            }

            if(totalSpeed < speedThreshold)
            {
                Debug.Log("Objects have stopped");
            }
        }
    }
}


