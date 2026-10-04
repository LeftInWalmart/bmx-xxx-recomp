// Export every instruction (with raw p-code semantics), the register file layout and all
// initialized memory bytes, so the sheets fully describe the executable.
// Usage: ghidra script run ExportPcode.java -- <outdir>
//@category Export
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.*;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.pcode.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;

public class ExportPcode extends GhidraScript {
    private String vn(Varnode v) {
        if (v == null) return "_";
        AddressSpace s = v.getAddress().getAddressSpace();
        char k;
        if (v.isConstant()) k = 'C';
        else if (v.isRegister()) k = 'R';
        else if (v.isUnique()) k = 'U';
        else if (s.isMemorySpace()) k = 'M';
        else k = 'X';
        return k + Long.toHexString(v.getOffset()) + ":" + v.getSize();
    }

    private String op(PcodeOp p) {
        StringBuilder b = new StringBuilder(p.getMnemonic());
        int start = 0;
        if (p.getOpcode() == PcodeOp.CALLOTHER) {
            b.append('[').append(currentProgram.getLanguage().getUserDefinedOpName((int) p.getInput(0).getOffset())).append(']');
            start = 1;
        } else if (p.getOpcode() == PcodeOp.LOAD || p.getOpcode() == PcodeOp.STORE) {
            start = 1; // input0 is the space id; everything here is in ram
        }
        b.append(' ').append(vn(p.getOutput())).append(" =");
        for (int i = start; i < p.getNumInputs(); i++) b.append(' ').append(vn(p.getInput(i)));
        return b.toString();
    }

    private static String hex8(long v) { return String.format("0x%08X", v); }

    @Override
    protected void run() throws Exception {
        Path out = Paths.get(getScriptArgs().length > 0 ? getScriptArgs()[0] : "export");
        Files.createDirectories(out);
        Listing listing = currentProgram.getListing();
        FunctionManager fm = currentProgram.getFunctionManager();

        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out.resolve("registers.csv"), StandardCharsets.UTF_8))) {
            w.println("name,offset,size");
            for (Register r : currentProgram.getLanguage().getRegisters())
                w.println(r.getName() + "," + hex8(r.getOffset()) + "," + r.getMinimumByteSize());
        }

        long n = 0;
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out.resolve("instructions.csv"), StandardCharsets.UTF_8))) {
            w.println("addr,func,bytes,asm,pcode");
            for (Instruction ins : listing.getInstructions(true)) {
                if (monitor.isCancelled()) break;
                Function f = fm.getFunctionContaining(ins.getAddress());
                StringBuilder pc = new StringBuilder();
                for (PcodeOp p : ins.getPcode()) {
                    if (pc.length() > 0) pc.append(" | ");
                    pc.append(op(p));
                }
                StringBuilder bytes = new StringBuilder();
                for (byte x : ins.getBytes()) bytes.append(String.format("%02X", x));
                String asm = ins.toString().replace("\"", "\"\"");
                w.println(hex8(ins.getAddress().getOffset()) + "," + (f == null ? "" : hex8(f.getEntryPoint().getOffset()))
                    + "," + bytes + ",\"" + asm + "\",\"" + pc + "\"");
                if (++n % 100000 == 0) println("instructions: " + n);
            }
        }

        long rows = 0;
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out.resolve("memory.csv"), StandardCharsets.UTF_8))) {
            w.println("addr,section,hex");
            byte[] buf = new byte[64];
            for (MemoryBlock b : currentProgram.getMemory().getBlocks()) {
                if (!b.isInitialized()) continue;
                long size = b.getSize();
                for (long o = 0; o < size; o += 64) {
                    int len = (int) Math.min(64, size - o);
                    b.getBytes(b.getStart().add(o), buf, 0, len);
                    StringBuilder h = new StringBuilder(len * 2);
                    for (int i = 0; i < len; i++) h.append(String.format("%02X", buf[i]));
                    w.println(hex8(b.getStart().getOffset() + o) + "," + b.getName() + "," + h);
                    rows++;
                }
            }
        }
        println("done: " + n + " instructions, " + rows + " memory rows");
    }
}
