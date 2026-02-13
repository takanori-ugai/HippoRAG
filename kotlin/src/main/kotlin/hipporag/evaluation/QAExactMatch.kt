package hipporag.evaluation

import hipporag.utils.normalizeAnswer

class QAExactMatch {
    fun calculateMetricScores(
        goldAnswers: List<List<String>>,
        predictedAnswers: List<String>,
        aggregationFn: (List<Double>) -> Double,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        require(goldAnswers.size == predictedAnswers.size) {
            "Length of gold answers and predicted answers should be the same."
        }

        val exampleEvalResults = mutableListOf<Map<String, Double>>()
        var totalEm = 0.0

        for ((goldList, predicted) in goldAnswers.zip(predictedAnswers)) {
            val emScores =
                goldList.map { gold ->
                    if (normalizeAnswer(gold) == normalizeAnswer(predicted)) 1.0 else 0.0
                }
            val aggregatedEm = aggregationFn(emScores)
            exampleEvalResults.add(mapOf("ExactMatch" to aggregatedEm))
            totalEm += aggregatedEm
        }

        val avgEm = if (goldAnswers.isNotEmpty()) totalEm / goldAnswers.size else 0.0
        val pooledEvalResults = mapOf("ExactMatch" to avgEm)

        return pooledEvalResults to exampleEvalResults
    }
}
