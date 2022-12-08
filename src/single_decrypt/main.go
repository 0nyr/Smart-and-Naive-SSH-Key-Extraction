// Single Decrypt
// Decrypts a single file using a key file

package main

import (
	"flag"
	"phdtrack/custom_log"

	_ "github.com/google/gopacket"
	_ "github.com/google/gopacket/pcap"
)

func main() {

	// flags
	verboseLevelFlag := flag.Int("v", 0, "verbosity level")

	flag.Parse()

	// overwrite global variables with flags
	custom_log.verboseLevel = *verboseLevelFlag

	custom_log.customLog(0, "Finished single decrypt")
	custom_log.cErr("This is an error")
	custom_log.cWarn("This is a warning")
	custom_log.cDebug("This is a debug message")
}
