# Vector Similarity Search Techniques in Faiss

This guide provides an in-depth explanation of the vector similarity search techniques available in Faiss. Whether you're building a recommendation system, semantic search engine, or image retrieval application, understanding these techniques will help you choose the right approach for your use case.

## Table of Contents

1. [Introduction to Vector Similarity Search](#introduction-to-vector-similarity-search)
2. [Distance Metrics](#distance-metrics)
3. [Exact Search (Flat Indexes)](#exact-search-flat-indexes)
4. [Inverted File Indexes (IVF)](#inverted-file-indexes-ivf)
5. [Product Quantization (PQ)](#product-quantization-pq)
6. [Scalar Quantization (SQ)](#scalar-quantization-sq)
7. [Hierarchical Navigable Small World (HNSW)](#hierarchical-navigable-small-world-hnsw)
8. [Locality Sensitive Hashing (LSH)](#locality-sensitive-hashing-lsh)
9. [Composite Indexes](#composite-indexes)
10. [GPU Acceleration](#gpu-acceleration)
11. [Choosing the Right Index](#choosing-the-right-index)
12. [Practical Examples](#practical-examples)

---

## Introduction to Vector Similarity Search

Vector similarity search is the task of finding vectors in a database that are most similar to a given query vector. This is fundamental to many modern AI applications:

- **Recommendation Systems**: Finding similar items or users based on embedding vectors
- **Semantic Search**: Finding documents with similar meaning using text embeddings
- **Image Retrieval**: Finding visually similar images using CNN feature vectors
- **Duplicate Detection**: Identifying near-duplicate content in large datasets
- **Clustering**: Grouping similar items together

### The Core Problem

Given:
- A database of `n` vectors, each of dimension `d`
- A query vector of dimension `d`
- A parameter `k` (number of nearest neighbors to return)

Find the `k` vectors in the database that are closest to the query vector according to some distance metric.

### Why Is This Hard?

For small datasets (thousands of vectors), you can simply compute the distance to every vector in the database (brute force). However, as datasets grow to millions or billions of vectors:

1. **Memory**: Storing all vectors in RAM becomes challenging
2. **Speed**: Computing distances to all vectors becomes too slow
3. **Trade-offs**: You need to balance accuracy vs. speed vs. memory usage

Faiss provides a variety of indexing structures that address these challenges through different trade-offs.

---

## Distance Metrics

Before diving into indexing structures, it's essential to understand the distance metrics Faiss supports.

### L2 Distance (Euclidean Distance)

The most common distance metric, measuring the straight-line distance between two points:

```
L2(a, b) = sqrt(sum((a_i - b_i)^2))
```

In Faiss, L2 squared is typically used (without the square root) for efficiency:

```
L2_squared(a, b) = sum((a_i - b_i)^2)
```

**Use cases**: When the absolute position in vector space matters. Common for image embeddings, geometric data.

### Inner Product (Dot Product)

Measures the alignment between two vectors:

```
IP(a, b) = sum(a_i * b_i)
```

**Note**: Faiss returns the **maximum** inner product for nearest neighbor search (unlike L2 which returns minimum distance).

**Use cases**: When direction matters more than magnitude. Common for recommendation systems, normalized embeddings.

### Cosine Similarity

Measures the angle between two vectors:

```
cosine(a, b) = IP(a, b) / (||a|| * ||b||)
```

In Faiss, cosine similarity is achieved by:
1. Normalizing all vectors to unit length
2. Using inner product search

**Use cases**: Text embeddings, when you want to ignore vector magnitude.

### Choosing a Metric

| Metric | When to Use | Faiss Constant |
|--------|-------------|----------------|
| L2 | Default choice, absolute distances matter | `METRIC_L2` |
| Inner Product | Dot product scoring, normalized vectors | `METRIC_INNER_PRODUCT` |
| Cosine | Angle-based similarity (normalize first) | Use `METRIC_INNER_PRODUCT` on normalized vectors |

---

## Exact Search (Flat Indexes)

Flat indexes perform exhaustive (brute-force) search by computing distances to all database vectors.

### IndexFlatL2

The simplest index that stores all vectors as-is and computes L2 distance to every vector during search.

```python
import faiss
import numpy as np

d = 128  # dimension
nb = 100000  # database size

# Generate random data
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((10, d)).astype('float32')  # 10 queries

# Create and populate the index
index = faiss.IndexFlatL2(d)
index.add(xb)

# Search for 4 nearest neighbors
k = 4
D, I = index.search(xq, k)
# D contains distances, I contains indices
```

### IndexFlatIP

Same as IndexFlatL2 but uses inner product instead of L2 distance.

```python
index = faiss.IndexFlatIP(d)
```

### Characteristics

| Property | Value |
|----------|-------|
| **Accuracy** | 100% (exact search) |
| **Search Speed** | O(n × d) - linear in dataset size |
| **Memory** | 4 × n × d bytes (float32) |
| **Training** | Not required |
| **Add/Remove** | Supports both |

### When to Use

- Dataset size < 1 million vectors
- Need exact results (no approximation acceptable)
- Dimension is relatively small (< 256)
- As a baseline for accuracy comparison

---

## Inverted File Indexes (IVF)

IVF indexes partition the vector space using clustering (k-means). During search, only a subset of partitions are examined, dramatically reducing the number of distance computations.

### How It Works

1. **Training Phase**:
   - Run k-means clustering on a sample of vectors
   - This creates `nlist` centroids (cluster centers)

2. **Adding Vectors**:
   - Each vector is assigned to its nearest centroid
   - Vectors are stored in "inverted lists" (one per centroid)

3. **Search Phase**:
   - Find the `nprobe` centroids nearest to the query
   - Only search vectors in those `nprobe` lists
   - Return the overall k-nearest neighbors

### IndexIVFFlat

Stores vectors exactly (like Flat) but uses IVF partitioning for faster search.

```python
import faiss

d = 128
nlist = 100  # number of clusters

# The quantizer determines how centroids are compared
quantizer = faiss.IndexFlatL2(d)

# Create the IVF index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# Must train before adding vectors
index.train(xb)  # train on some vectors
index.add(xb)    # add vectors

# Control accuracy/speed trade-off
index.nprobe = 10  # search 10 nearest clusters (default is 1)

D, I = index.search(xq, k)
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `nlist` | Number of clusters/partitions | sqrt(n) to n/1000 |
| `nprobe` | Number of clusters to search | 1 to nlist |

**Trade-off**: Higher `nprobe` = better accuracy but slower search.

### Characteristics

| Property | Value |
|----------|-------|
| **Accuracy** | High (depends on nprobe) |
| **Search Speed** | O(nprobe × n/nlist × d) |
| **Memory** | ~4 × n × d bytes + centroids |
| **Training** | Required (k-means) |

### When to Use

- Dataset size > 100,000 vectors
- Willing to trade small accuracy loss for speed
- Have representative training data available

---

## Product Quantization (PQ)

Product Quantization is a compression technique that drastically reduces memory usage while enabling fast approximate distance computation.

### How It Works

1. **Vector Decomposition**:
   - Split each d-dimensional vector into `m` sub-vectors
   - Each sub-vector has d/m dimensions

2. **Codebook Learning** (Training):
   - For each of the m sub-spaces, learn a codebook of 256 centroids (8 bits)
   - This requires running k-means m times

3. **Encoding**:
   - Each sub-vector is replaced by the index of its nearest centroid
   - A d-dimensional float32 vector becomes m bytes

4. **Distance Computation**:
   - Precompute distances from query to all centroids (lookup table)
   - Approximate distance = sum of lookup table values

### IndexPQ

Standalone PQ index (not combined with IVF).

```python
import faiss
import numpy as np

d = 128
nb = 100000  # database size
m = 16  # number of sub-vectors (must divide d evenly)

# Sample data
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((10, d)).astype('float32')  # queries

index = faiss.IndexPQ(d, m, 8)  # 8 bits per sub-vector
index.train(xb)
index.add(xb)

k = 4
D, I = index.search(xq, k)
```

### IndexIVFPQ

Combines IVF partitioning with PQ compression - one of the most popular index types.

```python
import faiss

d = 128
nlist = 100
m = 16

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

index.train(xb)
index.add(xb)
index.nprobe = 10

D, I = index.search(xq, k)
```

### Memory Comparison

For 1 million vectors of dimension 128:
- **IndexFlatL2**: 1M × 128 × 4 bytes = **512 MB**
- **IndexPQ (m=16)**: 1M × 16 bytes = **16 MB** (32x compression!)

### Key Parameters

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| `m` | Number of sub-vectors | Higher = more accurate, more memory |
| `nbits` | Bits per sub-quantizer | Usually 8 (256 centroids) |

### Characteristics

| Property | Value |
|----------|-------|
| **Accuracy** | Moderate (lossy compression) |
| **Search Speed** | Very fast (integer operations) |
| **Memory** | m bytes per vector |
| **Training** | Required |

### When to Use

- Memory is limited
- Dataset is very large (10M+ vectors)
- Willing to accept some accuracy loss
- Combined with IVF for best results

---

## Scalar Quantization (SQ)

Scalar Quantization is a simpler compression technique that quantizes each dimension independently.

### How It Works

1. For each dimension, compute the min and max values
2. Map each float value to an integer in a fixed range (e.g., 0-255 for 8-bit)
3. Store the quantized integers instead of floats

### IndexScalarQuantizer

```python
import faiss

d = 128
index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
index.train(xb)
index.add(xb)

D, I = index.search(xq, k)
```

### Available Quantization Types

| Type | Bits/Dimension | Compression Ratio |
|------|----------------|-------------------|
| `QT_8bit` | 8 | 4x |
| `QT_6bit` | 6 | ~5.3x |
| `QT_4bit` | 4 | 8x |
| `QT_fp16` | 16 | 2x |

### IndexIVFScalarQuantizer

Combines IVF with SQ for partitioned, compressed search.

```python
import faiss

d = 128
nlist = 100

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFScalarQuantizer(
    quantizer, d, nlist,
    faiss.ScalarQuantizer.QT_8bit
)
index.train(xb)
index.add(xb)
```

### Characteristics

| Property | Value |
|----------|-------|
| **Accuracy** | Better than PQ for same memory |
| **Search Speed** | Fast (but slower than PQ) |
| **Memory** | nbits/8 bytes per dimension |
| **Training** | Required (to compute ranges) |

### SQ vs PQ Comparison

| Aspect | Scalar Quantization | Product Quantization |
|--------|---------------------|----------------------|
| Compression | Moderate (4-8x) | High (up to 64x) |
| Accuracy | Better | Lower |
| Speed | Slower | Faster (lookup tables) |
| Best for | Mid-size datasets | Large datasets |

---

## Hierarchical Navigable Small World (HNSW)

HNSW is a graph-based index that provides excellent query performance with minimal accuracy loss.

### How It Works

HNSW builds a multi-layer graph structure:

1. **Graph Construction**:
   - Each vector is a node in the graph
   - Nodes are connected to their approximate nearest neighbors
   - Multiple layers with decreasing density (like a skip list)

2. **Search**:
   - Start from the top layer (sparse)
   - Greedily navigate toward the query vector
   - Move to lower layers (denser) as you get closer
   - Return the nearest neighbors found at the bottom layer

### IndexHNSW

```python
import faiss

d = 128
M = 32  # number of connections per node

index = faiss.IndexHNSWFlat(d, M)

# Optional: set construction parameters
index.hnsw.efConstruction = 40

# HNSW doesn't need training (but does need data)
index.add(xb)

# Set search parameter
index.hnsw.efSearch = 64  # larger = more accurate but slower

D, I = index.search(xq, k)
```

### Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `M` | Connections per node | 16-64 |
| `efConstruction` | Search breadth during construction | 40-500 |
| `efSearch` | Search breadth during query | 16-512 |

### Characteristics

| Property | Value |
|----------|-------|
| **Accuracy** | Very high (often 99%+) |
| **Search Speed** | Very fast (sub-linear) |
| **Memory** | ~1.5x raw data + graph overhead |
| **Training** | Not required (builds during add) |
| **Add/Remove** | Add is efficient, remove is costly |

### Pros and Cons

**Pros:**
- Excellent accuracy/speed trade-off
- No training required
- Works well across different dimensions

**Cons:**
- Higher memory overhead than PQ/SQ
- Cannot easily remove vectors
- Build time can be long for large datasets

### When to Use

- Need high accuracy with fast queries
- Memory is not the primary constraint
- Dataset is relatively static (few deletions)
- Dimension is moderate to high

---

## Locality Sensitive Hashing (LSH)

LSH uses random projections to hash similar vectors to the same buckets with high probability.

### How It Works

1. Generate random hyperplanes
2. Hash each vector based on which side of each hyperplane it falls
3. Similar vectors are likely to have the same hash
4. During search, find vectors with matching or similar hashes

### IndexLSH

```python
import faiss

d = 128
nbits = 256  # number of hash bits

index = faiss.IndexLSH(d, nbits)
index.train(xb)
index.add(xb)

D, I = index.search(xq, k)
```

### Characteristics

| Property | Value |
|----------|-------|
| **Accuracy** | Lower than other methods |
| **Search Speed** | Fast |
| **Memory** | Low (binary codes) |
| **Training** | Required (random projections) |

### When to Use

- Very large, high-dimensional datasets
- Approximate results are acceptable
- Memory is extremely limited

**Note**: In practice, PQ and HNSW often outperform LSH for most use cases.

---

## Composite Indexes

Faiss allows combining multiple techniques for optimal performance.

### The Index Factory

The `index_factory` function creates composite indexes using a string description:

```python
import faiss

d = 128

# Create different index types
index1 = faiss.index_factory(d, "Flat")
index2 = faiss.index_factory(d, "IVF100,Flat")
index3 = faiss.index_factory(d, "IVF100,PQ16")
index4 = faiss.index_factory(d, "HNSW32")
index5 = faiss.index_factory(d, "IVF100,SQ8")
```

### Index Factory String Syntax

| Component | Description | Example |
|-----------|-------------|---------|
| `Flat` | Exact search | `"Flat"` |
| `IVFx` | IVF with x clusters | `"IVF100,Flat"` |
| `PQx` | PQ with x sub-vectors | `"PQ16"` |
| `SQx` | Scalar quantization | `"SQ8"` |
| `HNSWx` | HNSW with x connections | `"HNSW32"` |
| `OPQx_y` | Optimized PQ preprocessing | `"OPQ16_64,IVF100,PQ16"` |

### Pre-processing Transforms

You can add pre-processing steps:

```python
# PCA to reduce dimension before indexing
index = faiss.index_factory(d, "PCA64,IVF100,Flat")

# OPQ (Optimized Product Quantization)
index = faiss.index_factory(d, "OPQ16,IVF100,PQ16")
```

### Popular Composite Indexes

| String | Description | Use Case |
|--------|-------------|----------|
| `"IVF100,Flat"` | Partitioned exact search | Medium datasets, high accuracy |
| `"IVF100,PQ16"` | Partitioned + compressed | Large datasets, memory constrained |
| `"HNSW32,Flat"` | Graph-based | High accuracy, fast queries |
| `"OPQ16,IVF100,PQ16"` | Optimized compression | Best accuracy for PQ |
| `"IVF100,SQ8"` | Partitioned + scalar quantized | Balance of accuracy and compression |

---

## GPU Acceleration

Faiss provides GPU implementations that can dramatically speed up both indexing and search operations.

### Basic GPU Usage

```python
import faiss
import numpy as np

d = 128
nb = 1000000
k = 4

# Sample data
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((10, d)).astype('float32')  # queries

# Create a CPU index
cpu_index = faiss.IndexFlatL2(d)

# Move to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # GPU 0

# Use like normal
gpu_index.add(xb)
D, I = gpu_index.search(xq, k)
```

### GPU-Native Indexes

Some indexes are designed specifically for GPU:

```python
import faiss

d = 128
nlist = 100  # number of clusters

res = faiss.StandardGpuResources()

# GPU-native flat index
index = faiss.GpuIndexFlatL2(res, d)

# GPU-native IVF index
index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)
```

### Multi-GPU Support

```python
# Use multiple GPUs
ngpus = faiss.get_num_gpus()

# Replicate index across all GPUs
cpu_index = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
```

### GPU vs CPU Performance

| Operation | GPU Speedup (typical) |
|-----------|----------------------|
| Brute-force search | 10-50x |
| IVF search | 5-20x |
| k-means training | 20-100x |

### When to Use GPU

- Very large-scale searches (millions of queries)
- Need lowest possible latency
- Have NVIDIA GPU(s) available
- Batch processing is acceptable

---

## Choosing the Right Index

### Decision Flowchart

```
Start
  │
  ├── Dataset < 10K vectors?
  │   └── Use IndexFlatL2/IP (exact search is fast enough)
  │
  ├── Need exact results?
  │   └── Use IndexFlatL2/IP with GPU if available
  │
  ├── Memory is primary constraint?
  │   ├── < 1GB available → Use IVF + PQ
  │   └── Otherwise → Use IVF + SQ8
  │
  ├── Query speed is primary constraint?
  │   └── Use HNSW or GPU-accelerated index
  │
  └── Balanced requirements?
      └── Use IVF + Flat or HNSW
```

### Index Comparison Summary

| Index | Search Speed | Memory | Accuracy | Training |
|-------|--------------|--------|----------|----------|
| Flat | Slow (O(n)) | High | Exact | No |
| IVFFlat | Medium | High | High | Yes |
| IVFPQ | Fast | Low | Medium | Yes |
| IVFSQ | Medium-Fast | Medium | High | Yes |
| HNSW | Fast | Medium-High | Very High | No |

### Rules of Thumb

1. **Start simple**: Begin with `IndexFlatL2` to establish baseline accuracy
2. **Measure first**: Profile your actual queries before optimizing
3. **Consider your constraints**: Memory? Speed? Accuracy?
4. **Tune parameters**: Most indexes have parameters that trade off accuracy vs. speed
5. **Use the index factory**: It simplifies creating and experimenting with different indexes

---

## Practical Examples

### Example 1: Image Search System

Building a reverse image search for 10 million images with 2048-dimensional CNN features:

```python
import faiss
import numpy as np

d = 2048
nb = 10_000_000

# Load your image features
# xb = load_image_features()  # shape: (nb, d)

# For large dataset with memory constraints, use IVF + PQ
nlist = 4096  # sqrt(10M) ≈ 3162, round up to power of 2
m = 64  # 2048/64 = 32-dim sub-vectors

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

# Train on a subset (1% is usually enough)
train_size = min(100000, nb)
train_data = xb[:train_size]
index.train(train_data)

# Add all vectors
index.add(xb)

# Search parameters for good accuracy
index.nprobe = 64

# Search
query_features = get_query_image_features()  # shape: (1, d)
k = 10
D, I = index.search(query_features, k)

# I contains indices of the 10 most similar images
```

### Example 2: Semantic Text Search

Building a semantic search engine with 768-dimensional sentence embeddings:

```python
import faiss
import numpy as np

d = 768
nb = 1_000_000

# For semantic search, we want high accuracy
# HNSW provides excellent accuracy/speed trade-off

index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = 200

# Add vectors (no training needed)
index.add(xb)

# Set high efSearch for better accuracy
index.hnsw.efSearch = 256

# Normalize query for cosine similarity
query = get_query_embedding()
faiss.normalize_L2(query)

k = 20
D, I = index.search(query, k)
```

### Example 3: Real-time Recommendations

Building a recommendation system requiring sub-millisecond latency:

```python
import faiss
import numpy as np

d = 128
nb = 5_000_000

# Use GPU for lowest latency
res = faiss.StandardGpuResources()

# Use IVF for fast search
nlist = 2048
quantizer = faiss.IndexFlatL2(d)

# Create GPU index directly
index = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2)

# Train and add
index.train(xb)
index.add(xb)

# Set nprobe for accuracy
index.nprobe = 32

# Batch queries are more efficient on GPU
batch_size = 1000
queries = get_user_queries(batch_size)  # shape: (1000, d)
k = 10
D, I = index.search(queries, k)
```

### Example 4: Billion-Scale Search

Handling 1 billion vectors that don't fit in RAM:

```python
import faiss
import numpy as np

d = 128
nb = 1_000_000_000  # 1 billion

# Strategy: On-disk IVF index
nlist = 65536  # Many clusters for efficiency

# Build the index in shards
shard_size = 10_000_000  # 10M per shard

# Create index factory string for OPQ + IVF + PQ
index = faiss.index_factory(d, "OPQ32,IVF65536,PQ32")

# Train on a sample
sample_size = 1_000_000
# training_sample = load_training_data(sample_size)  # Load your training data
training_sample = np.random.random((sample_size, d)).astype('float32')
index.train(training_sample)

# Add vectors in batches (can be distributed across machines)
for i in range(0, nb, shard_size):
    # xb_shard = load_shard(i, shard_size)  # Load your data shard
    xb_shard = np.random.random((shard_size, d)).astype('float32')
    index.add(xb_shard)

# Save to disk
faiss.write_index(index, "billion_scale.index")

# Load and search
index = faiss.read_index("billion_scale.index")
index.nprobe = 128

# Query vectors
xq = np.random.random((10, d)).astype('float32')
k = 10
D, I = index.search(xq, k)
```

---

## Further Reading

- [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki) - Comprehensive documentation
- [Faiss Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started) - Getting started guide
- [Faiss FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ) - Frequently asked questions
- [Faiss Paper](https://arxiv.org/abs/2401.08281) - Academic paper on Faiss
- [GPU Faiss Paper](https://arxiv.org/abs/1702.08734) - Billion-scale similarity search with GPUs

## Summary

Faiss provides a rich toolkit for vector similarity search at any scale:

| Technique | Best For |
|-----------|----------|
| **Flat** | Small datasets, exact search baseline |
| **IVF** | Medium to large datasets, partitioned search |
| **PQ** | Memory-constrained scenarios, lossy compression |
| **SQ** | Balance between accuracy and compression |
| **HNSW** | High accuracy with fast queries |
| **GPU** | Maximum throughput and lowest latency |

The key to success is understanding your specific requirements (accuracy, speed, memory, scale) and choosing the right combination of techniques. Start with a simple index, measure performance, and iterate towards more sophisticated solutions as needed.
