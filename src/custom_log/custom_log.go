package custom_log

import (
	"fmt"
	"log"
	"os"
)

// global variables
var (
	// 0: unimportant messages
	// 1: important messages (debug)
	// 2: debug messages (warnings)
	// 3: trace messages (erros and panics)
	VerboseLevel = 0 // verbosity level. 0 is the lowest (all messages), 3 is the highest (errors).
)

// Custom Logging function.
// logLevel is the level of verbosity. 0 is the lowest (no message), 3 is the highest.
func CustomLog(
	logLevel int,
	message string,
) {
	if logLevel >= VerboseLevel {
		if logLevel >= 2 {
			fmt.Fprintln(os.Stderr, message)
		}
		log.Default().Println(message)
	}
}

func CLog(message string) {
	CustomLog(0, message)
}

func CDebug(message string) {
	CustomLog(1, message)
}

func CWarn(message string) {
	message = "WARNING: " + message
	CustomLog(2, message)
}

func CErr(message string) {
	message = "ERROR: " + message
	CustomLog(3, message)
}
