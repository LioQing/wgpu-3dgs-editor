$BASE_DIR = $PSScriptRoot
$EXAMPLES_PATH = "$BASE_DIR/../examples"
$LCOV_PATH = "$BASE_DIR/lcov.info"

echo "Running coverage..."

cargo llvm-cov clean --workspace

echo "Running 'modify' example"
cargo llvm-cov run --example modify -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" --rgb-or-hsv 0.1 0.9 0.8 --alpha 0.7 --contrast "-1.0" --exposure 0.5 --gamma 1.2

echo "Running 'modify' example with '--override-rgb' flag"
cargo llvm-cov run --example modify -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" --override-rgb --rgb-or-hsv 0.7 0.9 0.8 --alpha 0.7 --contrast "-1.0" --exposure 0.5 --gamma 1.2

echo "Running 'modify-selection' example"
cargo llvm-cov run --example modify-selection -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --repeat 2 --offset 2.0 0.0 0.0 --contrast "-1.0"

echo "Running 'modify-selection' example with '--shape Box' flag"
cargo llvm-cov run --example modify-selection -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --shape Box --repeat 2 --offset 2.0 0.0 0.0 --contrast "-1.0"

echo "Running 'filter-selection' example"
cargo llvm-cov run --example filter-selection -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --repeat 2 --offset 2.0 0.0 0.0

echo "Running 'filter-selection' example with '--shape Box' flag"
cargo llvm-cov run --example filter-selection -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" -p 0.5 1.0 0.5 -r 0.2 0.0 0.0 1.0 -s 1 1 1 --shape Box --repeat 2 --offset 2.0 0.0 0.0

echo "Running 'custom-modify-selection' example"
cargo llvm-cov run --example custom-modify-selection -- -m "$EXAMPLES_PATH/model.ply" -o "$BASE_DIR/output.ply" -p 0.5 0.0 0.2 -r 2.0

# `--doctests` flag is currently unstable
# echo "Running doctests"
# cargo llvm-cov --no-report --doctests

echo "Running tests"
cargo llvm-cov --no-report nextest

echo "Generating coverage report"
cargo llvm-cov report --lcov --output-path "$LCOV_PATH"

echo "Generating badge"
$total = 0
$covered = 0
Select-String -Path "$LCOV_PATH" -Pattern "DA:" | ForEach-Object {
    if ($_ -match "DA:\d+,(\d+)") {
        $total++
        if ($matches[1] -ne "0") {
            $covered++
        }
    }
}

$badge_percentage = if ($total -eq 0) { 100 } else { [math]::Round(($covered / $total) * 100) }
$badge_color = if ($badge_percentage -ge 80) {
    "brightgreen"
} elseif ($badge_percentage -ge 50) {
    "yellow"
} else {
    "red"
}

"{
    `"schemaVersion`": 1,
    `"label`": `"coverage`",
    `"message`": `"$badge_percentage%`",
    `"color`": `"$badge_color`"
}" | Out-File -FilePath "$BASE_DIR/badge.json" -Encoding ascii

echo "Cleaning up"
rm "$BASE_DIR/output.ply"

echo "Done"