// The executable of the small app that `MacOS.install_autostart` builds (agent/platforms/macos.py).
//
// launchd runs this app instead of Python. The app has NSMicrophoneUsageDescription in its Info.plist, asks for the
// Microphone permission once (AVCaptureDevice.requestAccess), and runs the command in its arguments as a child.
// macOS checks the Microphone permission of this app for the child, so the daemon hears real audio.
// The child starts at once and does not wait for the answer, so a login with nobody at the Mac still starts the daemon.
// Signals that end the app (logout, `launchctl bootout`) go to the child. The app exits with the child's exit code,
// or 128 + the signal number when the child was killed, so launchd restarts the daemon after a crash only.
import AVFoundation
import Foundation

func log(_ line: String) {
    FileHandle.standardError.write("launcher: \(line)\n".data(using: .utf8)!)
}

let names: [AVAuthorizationStatus: String] = [
    .notDetermined: "not determined", .restricted: "restricted", .denied: "denied", .authorized: "authorized",
]
let status = AVCaptureDevice.authorizationStatus(for: .audio)
log("microphone permission \(names[status] ?? "unknown")")
if status == .notDetermined {
    AVCaptureDevice.requestAccess(for: .audio) { granted in
        log("microphone permission \(granted ? "authorized" : "denied") in the dialog")
    }
}

let args = Array(CommandLine.arguments.dropFirst())
guard let program = args.first else {
    log("no command to run")
    exit(64)
}
let child = Process()
child.executableURL = URL(fileURLWithPath: program)
child.arguments = Array(args.dropFirst())
var sources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT, SIGHUP] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { kill(child.processIdentifier, sig) }
    source.resume()
    sources.append(source)
}
child.terminationHandler = { p in
    exit(p.terminationReason == .uncaughtSignal ? 128 + p.terminationStatus : p.terminationStatus)
}
do {
    try child.run()
} catch {
    log("cannot run \(program): \(error)")
    exit(70)
}
dispatchMain()
