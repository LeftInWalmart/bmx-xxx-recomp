// Bulk-export the analyzed program into Spreadsheet-Method raw tables + per-function C.
// Usage: ghidra script run ExportSheets.java -- <outdir>
//@category Export
import ghidra.app.decompiler.*;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.*;
import ghidra.program.model.data.StringDataInstance;
import ghidra.program.model.listing.*;
import ghidra.program.model.mem.MemoryBlock;
import ghidra.program.model.symbol.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class ExportSheets extends GhidraScript {
    private static String hex(Address a) { return String.format("0x%08X", a.getOffset()); }

    private static String q(String s) {
        if (s == null) return "";
        if (s.contains(",") || s.contains("\"") || s.contains("\n") || s.contains("\r"))
            return "\"" + s.replace("\"", "\"\"").replace("\r", "\\r").replace("\n", "\\n") + "\"";
        return s;
    }

    private PrintWriter open(Path p) throws IOException {
        return new PrintWriter(Files.newBufferedWriter(p, StandardCharsets.UTF_8));
    }

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        Path out = Paths.get(args.length > 0 ? args[0] : "export");
        Path raw = out.resolve("raw_c");
        Files.createDirectories(raw);

        Listing listing = currentProgram.getListing();
        ReferenceManager refs = currentProgram.getReferenceManager();
        FunctionManager fm = currentProgram.getFunctionManager();

        try (PrintWriter w = open(out.resolve("sections.csv"))) {
            w.println("name,start,end,size,perms,initialized");
            for (MemoryBlock b : currentProgram.getMemory().getBlocks())
                w.println(String.join(",", q(b.getName()), hex(b.getStart()), hex(b.getEnd()),
                    Long.toString(b.getSize()),
                    (b.isRead() ? "r" : "-") + (b.isWrite() ? "w" : "-") + (b.isExecute() ? "x" : "-"),
                    Boolean.toString(b.isInitialized())));
        }

        // Strings, plus which functions reference them.
        try (PrintWriter w = open(out.resolve("strings.csv"))) {
            w.println("addr,section,len,text,ref_funcs");
            for (Data d : listing.getDefinedData(true)) {
                if (monitor.isCancelled()) break;
                if (!StringDataInstance.isString(d)) continue;
                Object v = d.getValue();
                if (v == null) continue;
                TreeSet<String> fs = new TreeSet<>();
                for (Reference r : refs.getReferencesTo(d.getAddress())) {
                    Function f = fm.getFunctionContaining(r.getFromAddress());
                    if (f != null) fs.add(hex(f.getEntryPoint()));
                }
                MemoryBlock b = currentProgram.getMemory().getBlock(d.getAddress());
                w.println(String.join(",", hex(d.getAddress()), q(b == null ? "" : b.getName()),
                    Integer.toString(d.getLength()), q(v.toString()), String.join(";", fs)));
            }
        }

        // Global data symbols (non-function labels in non-exec blocks).
        try (PrintWriter w = open(out.resolve("globals.csv"))) {
            w.println("addr,name,section,datatype,size,xrefs");
            SymbolIterator it = currentProgram.getSymbolTable().getAllSymbols(true);
            for (Symbol s : it) {
                if (monitor.isCancelled()) break;
                if (s.getSymbolType() != SymbolType.LABEL || s.isExternal()) continue;
                MemoryBlock b = currentProgram.getMemory().getBlock(s.getAddress());
                if (b == null || b.isExecute()) continue;
                Data d = listing.getDataAt(s.getAddress());
                w.println(String.join(",", hex(s.getAddress()), q(s.getName()), q(b.getName()),
                    q(d == null ? "" : d.getDataType().getName()),
                    d == null ? "" : Integer.toString(d.getLength()),
                    Integer.toString(refs.getReferenceCountTo(s.getAddress()))));
            }
        }

        DecompInterface di = new DecompInterface();
        DecompileOptions opts = new DecompileOptions();
        opts.grabFromProgram(currentProgram);
        di.setOptions(opts);
        di.toggleCCode(true);
        di.toggleSyntaxTree(false);
        di.setSimplificationStyle("decompile");
        if (!di.openProgram(currentProgram)) throw new IllegalStateException(di.getLastMessage());

        int total = fm.getFunctionCount(), n = 0, failed = 0;
        monitor.initialize(total);
        try (PrintWriter fw = open(out.resolve("functions_raw.csv"));
             PrintWriter cw = open(out.resolve("calls.csv"))) {
            fw.println("addr,name,size,section,signature,calls_out,called_by,decomp_lines,decomp_ok");
            cw.println("caller,callee");
            for (Function f : fm.getFunctions(true)) {
                if (monitor.isCancelled()) break;
                monitor.setProgress(++n);
                if (f.isExternal() || f.isThunk() && f.getBody().getNumAddresses() == 0) continue;
                Address ep = f.getEntryPoint();
                Set<Function> callees = f.getCalledFunctions(monitor);
                Set<Function> callers = f.getCallingFunctions(monitor);
                for (Function c : callees) cw.println(hex(ep) + "," + hex(c.getEntryPoint()));

                String code;
                boolean ok;
                DecompileResults res = di.decompileFunction(f, 60, monitor);
                if (res != null && res.decompileCompleted() && res.getDecompiledFunction() != null) {
                    code = res.getDecompiledFunction().getC();
                    ok = true;
                } else {
                    code = "// decompile failed: " + (res == null ? "null" : res.getErrorMessage()) + "\n";
                    ok = false;
                    failed++;
                }
                String fname = String.format("%08X.c", ep.getOffset());
                Files.writeString(raw.resolve(fname),
                    "// " + hex(ep) + " " + f.getName() + " size=" + f.getBody().getNumAddresses() + "\n" + code,
                    StandardCharsets.UTF_8);
                MemoryBlock b = currentProgram.getMemory().getBlock(ep);
                fw.println(String.join(",", hex(ep), q(f.getName()),
                    Long.toString(f.getBody().getNumAddresses()), q(b == null ? "" : b.getName()),
                    q(f.getSignature().getPrototypeString()),
                    Integer.toString(callees.size()), Integer.toString(callers.size()),
                    Long.toString(code.chars().filter(c -> c == '\n').count()), Boolean.toString(ok)));
                if (n % 500 == 0) println("decompiled " + n + "/" + total);
            }
        } finally {
            di.dispose();
        }
        println("done: " + n + " functions, " + failed + " failed -> " + out.toAbsolutePath());
    }
}
