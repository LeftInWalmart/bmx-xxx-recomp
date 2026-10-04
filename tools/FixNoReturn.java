// Undo wrong "non-returning" verdicts from Ghidra's auto-analysis. A function marked no-return
// that contains a return instruction (e.g. OSReport in BMX XXX) makes Ghidra stop disassembling
// after every call to it, so the rest of each caller (OSInit's tail, ...) is missing.
// For each such function: clear the flag, clear fall-through overrides at its call sites,
// disassemble the fall-through and repair the callers' bodies.
// Usage: ghidra script run FixNoReturn.java
//@category Analysis
import ghidra.app.cmd.disassemble.DisassembleCommand;
import ghidra.app.cmd.function.CreateFunctionCmd;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.symbol.*;

import java.util.*;

public class FixNoReturn extends GhidraScript {
    @Override
    protected void run() throws Exception {
        Listing listing = currentProgram.getListing();
        FunctionManager fm = currentProgram.getFunctionManager();
        ReferenceManager refs = currentProgram.getReferenceManager();
        int fixedFuncs = 0, fixedSites = 0;
        for (Function f : fm.getFunctions(true)) {
            if (monitor.isCancelled()) break;
            if (!f.hasNoReturn()) continue;
            boolean returns = false;
            for (Instruction ins : listing.getInstructions(f.getBody(), true)) {
                if (ins.getFlowType().isTerminal() && ins.getMnemonicString().startsWith("blr")) {
                    returns = true;
                    break;
                }
            }
            if (!returns) continue;
            f.setNoReturn(false);
            fixedFuncs++;
            println("returns after all: " + f.getName() + " @ " + f.getEntryPoint());
            Set<Function> callers = new HashSet<>();
            for (Reference r : refs.getReferencesTo(f.getEntryPoint())) {
                if (!r.getReferenceType().isCall()) continue;
                Instruction call = listing.getInstructionAt(r.getFromAddress());
                if (call == null) continue;
                call.clearFallThroughOverride();
                call.setFlowOverride(FlowOverride.NONE);
                Address next = call.getMaxAddress().add(1);
                if (listing.getInstructionAt(next) == null) {
                    new DisassembleCommand(next, null, true).applyTo(currentProgram, monitor);
                    fixedSites++;
                }
                Function caller = fm.getFunctionContaining(call.getAddress());
                if (caller != null) callers.add(caller);
            }
            for (Function caller : callers) {
                CreateFunctionCmd.fixupFunctionBody(currentProgram, caller, monitor);
            }
        }
        println("done: " + fixedFuncs + " functions no longer no-return, " + fixedSites + " call sites re-disassembled");
    }
}
