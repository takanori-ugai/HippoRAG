package hipporag.evaluation

import hipporag.utils.normalizeAnswer

class QAF1Score {
    fun calculateMetricScores(
        goldAnswers: List<List<String>>,
        predictedAnswers: List<String>,
        aggregationFn: (List<Double>) -> Double,
    ): Pair<Map<String, Double>, List<Map<String, Double>>> {
        require(goldAnswers.size == predictedAnswers.size) {
            "Length of gold answers and predicted answers should be the same."
        }

        fun computeF1(
            gold: String,
            predicted: String,
        ): Double {
            val goldTokens = normalizeAnswer(gold).split(" ").filter { it.isNotEmpty() }
            val predictedTokens = normalizeAnswer(predicted).split(" ").filter { it.isNotEmpty() }

            val goldCounts = goldTokens.groupingBy { it }.eachCount()
            val predictedCounts = predictedTokens.groupingBy { it }.eachCount()

            var numSame = 0
            for ((token, count) in predictedCounts) {
                val goldCount = goldCounts[token] ?: 0
                numSame += minOf(count, goldCount)
            }

            if (numSame == 0) return 0.0

            val precision = numSame.toDouble() / predictedTokens.size.toDouble()
            val recall = numSame.toDouble() / goldTokens.size.toDouble()
            return 2 * (precision * recall) / (precision + recall)
        }

        val exampleEvalResults = mutableListOf<Map<String, Double>>()
        var totalF1 = 0.0

        for ((goldList, predicted) in goldAnswers.zip(predictedAnswers)) {
            val f1Scores = goldList.map { gold -> computeF1(gold, predicted) }
            val aggregatedF1 = aggregationFn(f1Scores)
            exampleEvalResults.add(mapOf("F1" to aggregatedF1))
            totalF1 += aggregatedF1
        }

        val avgF1 = if (goldAnswers.isNotEmpty()) totalF1 / goldAnswers.size else 0.0
        val pooledEvalResults = mapOf("F1" to avgF1)

        return pooledEvalResults to exampleEvalResults
    }
}
