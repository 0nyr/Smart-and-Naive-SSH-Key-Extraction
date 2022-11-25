// Single Decrypt
// Decrypts a single file using a key file

package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	_ "github.com/google/gopacket"
	_ "github.com/google/gopacket/pcap"
)

// global variables
var (
	// 0: unimportant messages
	// 1: important messages (debug)
	// 2: debug messages (warnings)
	// 3: trace messages (erros and panics)
	verboseLevel = 0 // verbosity level. 0 is the lowest (all messages), 3 is the highest (errors).
)

// Custom Logging function.
// logLevel is the level of verbosity. 0 is the lowest (no message), 3 is the highest.
func customLog(
	logLevel int,
	message string,
) {
	if logLevel >= verboseLevel {
		if logLevel >= 2 {
			fmt.Fprintln(os.Stderr, message)
		}
		log.Default().Println(message)
	}
}

func cLog(message string) {
	customLog(0, message)
}

func cDebug(message string) {
	customLog(1, message)
}

func cWarn(message string) {
	message = "WARNING: " + message
	customLog(2, message)
}

func cErr(message string) {
	message = "ERROR: " + message
	customLog(3, message)
}

func main() {

	// flags
	verboseLevelFlag := flag.Int("v", 0, "verbosity level")

	flag.Parse()

	// overwrite global variables with flags
	verboseLevel = *verboseLevelFlag

	customLog(0, "Finished single decrypt")
	cErr("This is an error")
	cWarn("This is a warning")
	cDebug("This is a debug message")

}
