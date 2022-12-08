module single_decrypt

go 1.19

replace phdtrack/custom_log => ../custom_log

require (
	github.com/google/gopacket v1.1.19
	phdtrack/custom_log v0.0.0-00010101000000-000000000000
)

require golang.org/x/sys v0.0.0-20190412213103-97732733099d // indirect
