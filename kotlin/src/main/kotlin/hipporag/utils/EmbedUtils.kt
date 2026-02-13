package hipporag.utils

import smile.neighbor.LSH
import kotlin.math.sqrt

fun retrieveKnn(
    queryIds: List<String>,
    keyIds: List<String>,
    queryVecs: Array<DoubleArray>,
    keyVecs: Array<DoubleArray>,
    k: Int,
): Map<String, Pair<List<String>, List<Double>>> {
    if (keyVecs.isEmpty() || queryVecs.isEmpty()) return emptyMap()

    val normalizedKeys = keyVecs.map { normalize(it) }.toTypedArray()
    val normalizedQueries = queryVecs.map { normalize(it) }.toTypedArray()

    val keyData = keyIds.toTypedArray()

    val w = 4.0
    val lsh = LSH(normalizedKeys, keyData, w)

    val results = mutableMapOf<String, Pair<List<String>, List<Double>>>()

    for ((idx, query) in normalizedQueries.withIndex()) {
        val queryId = queryIds[idx]
        val neighbors = lsh.search(query, k)
        val neighborIds = mutableListOf<String>()
        val neighborScores = mutableListOf<Double>()

        for (neighbor in neighbors) {
            val keyId = neighbor.value()
            val distance = neighbor.distance()
            val similarity = 1.0 - (distance * distance) / 2.0
            neighborIds.add(keyId)
            neighborScores.add(similarity)
        }

        results[queryId] = neighborIds to neighborScores
    }

    return results
}

private fun normalize(vector: DoubleArray): DoubleArray {
    var sumSquares = 0.0
    for (v in vector) {
        sumSquares += v * v
    }
    val norm = sqrt(sumSquares)
    if (norm == 0.0) return vector.copyOf()
    return DoubleArray(vector.size) { idx -> vector[idx] / norm }
}
