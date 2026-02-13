package hipporag.utils

import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging

fun getLogger(name: String): KLogger = KotlinLogging.logger(name)
