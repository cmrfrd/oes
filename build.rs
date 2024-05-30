use std::process::{Command, Stdio};

fn build_stubs() {
    println!("cargo:rerun-if-changed=configs/openai_openapi.yaml");
    println!("cargo:rerun-if-changed=scripts/build_stubs.sh");

    let output = Command::new("bash")
        .arg("./scripts/build_stubs.sh")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("Failed to run build_stubs.sh");

    if !output.status.success() {
        println!(
            "cargo:warning=build_stubs.sh failed with exit code {:?}",
            output.status.code()
        );
        println!(
            "cargo:warning=stdout: {}",
            String::from_utf8_lossy(&output.stdout)
        );
        println!(
            "cargo:warning=stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn main() {
    build_stubs();
}
