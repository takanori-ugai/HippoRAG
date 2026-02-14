package hipporag.graph

import hipporag.utils.jsonWithDefaults
import kotlinx.serialization.Serializable
import java.io.File

/**
 * Lightweight in-memory graph with serialization support.
 *
 * @param directed whether edges are directed.
 */
class SimpleGraph(
    private val directed: Boolean,
) {
    private val vertices = mutableListOf<MutableMap<String, Any>>()
    private val edges = mutableListOf<Edge>()
    private val nameToIndex = mutableMapOf<String, Int>()

    /** Returns the number of vertices. */
    fun vcount(): Int = vertices.size

    /** Returns the number of edges. */
    fun ecount(): Int = edges.size

    /** Returns the list of vertex names (if present). */
    fun vertexNames(): List<String> = vertices.mapNotNull { it["name"]?.toString() }

    /**
     * Adds vertices using columnar [attributes], where each list element is a vertex attribute value.
     */
    fun addVertices(attributes: Map<String, List<Any>>) {
        if (attributes.isEmpty()) return
        val count = attributes.values.first().size
        require(attributes.values.all { it.size == count }) {
            "All attribute lists must have the same length ($count)"
        }
        for (i in 0 until count) {
            val attr = mutableMapOf<String, Any>()
            for ((key, values) in attributes) {
                attr[key] = values[i]
            }
            val idx = vertices.size
            vertices.add(attr)
            val name = attr["name"]?.toString()
            if (name != null) {
                nameToIndex[name] = idx
            }
        }
    }

    /**
     * Adds edges between named vertices with the supplied [weights].
     */
    fun addEdges(
        edgePairs: List<Pair<String, String>>,
        weights: List<Double>,
    ) {
        edgePairs.zip(weights).forEach { (pair, weight) ->
            val sourceIdx = nameToIndex[pair.first]
            val targetIdx = nameToIndex[pair.second]
            if (sourceIdx != null && targetIdx != null) {
                edges.add(Edge(sourceIdx, targetIdx, weight))
            }
        }
    }

    /**
     * Deletes vertices by name, removing associated edges.
     */
    fun deleteVertices(names: List<String>) {
        if (names.isEmpty()) return
        val removeSet = names.toSet()
        val survivingEdges =
            edges.mapNotNull { edge ->
                val sourceName = vertices.getOrNull(edge.source)?.get("name")?.toString()
                val targetName = vertices.getOrNull(edge.target)?.get("name")?.toString()
                if (sourceName == null || targetName == null) {
                    null
                } else if (sourceName in removeSet || targetName in removeSet) {
                    null
                } else {
                    Triple(sourceName, targetName, edge.weight)
                }
            }

        vertices.removeAll { it["name"]?.toString() in removeSet }
        edges.clear()
        nameToIndex.clear()
        vertices.forEachIndexed { idx, attrs ->
            val name = attrs["name"]?.toString()
            if (name != null) {
                nameToIndex[name] = idx
            }
        }

        for ((sourceName, targetName, weight) in survivingEdges) {
            val sourceIdx = nameToIndex[sourceName]
            val targetIdx = nameToIndex[targetName]
            if (sourceIdx != null && targetIdx != null) {
                edges.add(Edge(sourceIdx, targetIdx, weight))
            }
        }
    }

    /**
     * Computes personalized PageRank scores for all vertices.
     *
     * @param reset per-vertex reset probabilities (unnormalized).
     * @param damping damping factor.
     */
    fun personalizedPageRank(
        reset: DoubleArray,
        damping: Double,
    ): DoubleArray {
        val n = vertices.size
        if (n == 0) return DoubleArray(0)

        val resetSum = reset.sum()
        val resetProb = if (resetSum > 0) reset.map { it / resetSum }.toDoubleArray() else DoubleArray(n) { 1.0 / n }

        val adjacency = Array(n) { mutableListOf<Pair<Int, Double>>() }
        for (edge in edges) {
            adjacency[edge.source].add(edge.target to edge.weight)
            if (!directed) {
                adjacency[edge.target].add(edge.source to edge.weight)
            }
        }

        val outWeight = DoubleArray(n)
        for (i in 0 until n) {
            outWeight[i] = adjacency[i].sumOf { it.second }
        }

        var scores = DoubleArray(n) { 1.0 / n }
        val maxIter = 100
        val tol = 1e-6

        repeat(maxIter) {
            val next = DoubleArray(n) { (1.0 - damping) * resetProb[it] }
            for (i in 0 until n) {
                val weightSum = outWeight[i]
                if (weightSum == 0.0) continue
                val contribution = damping * scores[i] / weightSum
                for ((j, w) in adjacency[i]) {
                    next[j] += contribution * w
                }
            }

            val delta = next.zip(scores).sumOf { (a, b) -> kotlin.math.abs(a - b) }
            scores = next
            if (delta < tol) return scores
        }

        return scores
    }

    /**
     * Serializes the graph into [file].
     */
    fun save(file: File) {
        val data =
            GraphData(
                directed = directed,
                vertices = vertices.map { it.mapValues { v -> v.value.toString() } },
                edges = edges.map { EdgeData(it.source, it.target, it.weight) },
            )
        val json = jsonWithDefaults { prettyPrint = false }
        file.writeText(json.encodeToString(GraphData.serializer(), data))
    }

    companion object {
        /**
         * Loads a [SimpleGraph] from [file].
         */
        fun load(file: File): SimpleGraph {
            val json = jsonWithDefaults { ignoreUnknownKeys = true }
            val data = json.decodeFromString(GraphData.serializer(), file.readText())
            val graph = SimpleGraph(data.directed)
            val attributes = mutableMapOf<String, MutableList<Any>>()
            if (data.vertices.isNotEmpty()) {
                val keys = data.vertices.first().keys
                for (k in keys) {
                    attributes[k] = mutableListOf()
                }
                for (vertex in data.vertices) {
                    for ((k, v) in vertex) {
                        attributes.getValue(k).add(v)
                    }
                }
                graph.addVertices(attributes)
            }
            if (data.edges.isNotEmpty()) {
                val edgePairs =
                    data.edges.map { edge ->
                        val sourceName = graph.vertices[edge.source]["name"]?.toString() ?: ""
                        val targetName = graph.vertices[edge.target]["name"]?.toString() ?: ""
                        sourceName to targetName
                    }
                val weights = data.edges.map { it.weight }
                graph.addEdges(edgePairs, weights)
            }
            return graph
        }
    }

    private data class Edge(
        val source: Int,
        val target: Int,
        val weight: Double,
    )
}

/**
 * Serialized graph container.
 *
 * @property directed whether the graph is directed.
 * @property vertices vertex attribute maps.
 * @property edges edge list.
 */
@Serializable
data class GraphData(
    val directed: Boolean,
    val vertices: List<Map<String, String>>,
    val edges: List<EdgeData>,
)

/**
 * Serialized edge record.
 *
 * @property source source vertex index.
 * @property target target vertex index.
 * @property weight edge weight.
 */
@Serializable
data class EdgeData(
    val source: Int,
    val target: Int,
    val weight: Double,
)
