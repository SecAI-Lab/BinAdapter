from ghidra.app.script import GhidraScript

setAnalysisOption(currentProgram, "ASCII Strings", "false")
setAnalysisOption(currentProgram, "Apply Data Archives", "false")
setAnalysisOption(currentProgram, "Decompiler Switch Analysis", "false")
setAnalysisOption(currentProgram, "Decompiler Parameter ID", "false")
setAnalysisOption(currentProgram, "Embedded Media", "false")
setAnalysisOption(currentProgram, "External Entry References", "false")
setAnalysisOption(currentProgram, "Scalar Operand References", "false")
setAnalysisOption(currentProgram, "Shared Return Calls", "false")
setAnalysisOption(currentProgram, "Stack", "false")