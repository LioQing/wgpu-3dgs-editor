macro_rules! cargo {
    ($arg:literal $(, $fmt_args:expr),* $(,)?) => {{
        let cmd_arg = format!($arg, $($fmt_args),*);
        let args = cmd_arg.split_whitespace();

        println!("cargo {}", cmd_arg);

        let status = std::process::Command::new("cargo")
            .args(args)
            .status()
            .expect("failed to execute process");

        assert!(status.success(), "command 'cargo {}' failed", cmd_arg);
    }};
}

fn main() {
    let exe_path = std::env::current_exe().expect("current exe");

    let mut manifest_path = exe_path.parent().expect("exe parent").to_path_buf();
    while std::fs::read_dir(&manifest_path)
        .expect("read dir")
        .find(|entry| entry.as_ref().expect("entry").file_name() == "Cargo.toml")
        .is_none()
    {
        manifest_path = manifest_path.parent().expect("parent").to_path_buf();
    }

    let coverage_path = manifest_path.join("coverage");
    let examples_path = manifest_path.join("examples");
    let lcov_path = coverage_path.join("lcov.info");
    let lcov_path_str = lcov_path.to_str().expect("lcov path");
    let badge_path = coverage_path.join("badge.json");
    let model_path = examples_path.join("model.ply");
    let model_path_str = model_path.to_str().expect("model path");
    let output_path = coverage_path.join("output.ply");
    let output_path_str = output_path.to_str().expect("output path");

    println!("Running coverage...");

    cargo!("llvm-cov clean --workspace");

    println!("Running 'modify' example");
    cargo!(
        "llvm-cov run --example modify -- -m {model_path_str} -o {output_path_str} --rgb-or-hsv 0.1 0.9 0.8 --alpha 0.7 --contrast -1.0 --exposure 0.5 --gamma 1.2"
    );

    println!("Running 'modify' example with '--override-rgb' flag");
    cargo!(
        "llvm-cov run --example modify -- -m {model_path_str} -o {output_path_str} --override-rgb --rgb-or-hsv 0.7 0.9 0.8 --alpha 0.7 --contrast -1.0 --exposure 0.5 --gamma 1.2"
    );

    println!("Running 'modify-selection' example");
    cargo!(
        "llvm-cov run --example modify-selection -- -m {model_path_str} -o {output_path_str} -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --repeat 2 --offset 2.0 0.0 0.0 --contrast -1.0"
    );

    println!("Running 'modify-selection' example with '--shape Box' flag");
    cargo!(
        "llvm-cov run --example modify-selection -- -m {model_path_str} -o {output_path_str} -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --shape Box --repeat 2 --offset 2.0 0.0 0.0 --contrast -1.0"
    );

    println!("Running 'filter-selection' example");
    cargo!(
        "llvm-cov run --example filter-selection -- -m {model_path_str} -o {output_path_str} -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --repeat 2 --offset 2.0 0.0 0.0"
    );

    println!("Running 'filter-selection' example with '--shape Box' flag");
    cargo!(
        "llvm-cov run --example filter-selection -- -m {model_path_str} -o {output_path_str} -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --shape Box --repeat 2 --offset 2.0 0.0 0.0"
    );

    println!("Running 'custom-modify-selection' example");
    cargo!(
        "llvm-cov run --example custom-modify-selection -- -m {model_path_str} -o {output_path_str} -p 0.5 0.0 0.2 -r 2.0"
    );

    println!("Running doctests");
    // `--doctests` flag is currently unstable
    // cargo!("llvm-cov --no-report --doctests");
    cargo!("test --doc");

    println!("Running tests");
    cargo!("llvm-cov --no-report nextest --all-features");

    println!("Generating coverage report");
    cargo!(
        "llvm-cov report --lcov --ignore-filename-regex wgpu-3dgs-core --output-path {lcov_path_str}"
    );

    println!("Generating badge");

    let lcov = std::fs::read_to_string(&lcov_path).expect("read lcov.info");
    let mut total: u64 = 0;
    let mut covered: u64 = 0;

    for line in lcov.lines() {
        if !line.starts_with("DA:") {
            continue;
        }

        let mut parts = line[3..].split(',');
        let _line_number = parts.next();
        let hits_str = parts.next();

        let Some(hits_str) = hits_str else {
            continue;
        };
        let Ok(hits) = hits_str.parse::<u64>() else {
            continue;
        };

        total += 1;
        if hits != 0 {
            covered += 1;
        }
    }

    let badge_percentage: u64 = if total == 0 {
        100
    } else {
        ((covered as f32 / total as f32) * 100.0).round() as u64
    };

    let badge_color = if badge_percentage >= 80 {
        "brightgreen"
    } else if badge_percentage >= 50 {
        "yellow"
    } else {
        "red"
    };

    let badge_json = format!(
        r#"
{{
    "schemaVersion": 1,
    "label": "coverage",
    "message": "{badge_percentage}%",
    "color": "{badge_color}"
}}
        "#
    );
    std::fs::write(&badge_path, badge_json.trim().to_owned() + "\n").expect("write badge.json");

    println!("Cleaning up");
    std::fs::remove_file(&output_path).expect("remove output.ply");

    println!("Done");
}
