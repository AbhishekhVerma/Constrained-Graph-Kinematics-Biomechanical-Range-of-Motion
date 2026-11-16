#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

show_menu() {
    clear
    echo "=============================================================================="
    echo "  CONSTRAINED GRAPH KINEMATICS - LIVE DEMO LAUNCHER (Nov 17, 2026)"
    echo "  Foundations of Data Science Final Project Presentation"
    echo "=============================================================================="
    echo ""
    echo "  [1] Instant Offline Replay Demo (50% Tool Occlusion, Guaranteed Zero-Wi-Fi)"
    echo "  [2] Live Webcam / Continuity Camera Demo (720p HD, Press T for Tool Box)"
    echo "  [3] Live iPhone ARKit UDP Telemetry Stream (Port 5005 + 30Hz Fallback)"
    echo "  [4] Extreme 70% Tool Occlusion Stress Test (15/21 Joints Masked)"
    echo "  [5] Run Full Kinematics & Model Architecture Verification (7/7 Unit Tests)"
    echo "  [6] Open Presentation Slides (Final_Presentation_Slides.pdf)"
    echo "  [7] Open IEEE Project Report (Final_Project_Report.pdf)"
    echo "  [8] Open Examiner 1-Page Cheat Sheet (docs/EXAMINER_CHEATSHEET.md)"
    echo "  [9] Open Slide-by-Slide Speaker Script (docs/Slide_By_Slide_Presentation_Guide.md)"
    echo "  [Q] Quit"
    echo ""
    echo "=============================================================================="
}

while true; do
    show_menu
    read -rp "Enter choice [1-9, default=1, Q to quit]: " CHOICE
    CHOICE="${CHOICE:-1}"

    case "$CHOICE" in
        1)
            echo ""
            echo "[*] Launching Offline Replay Demo (50% Occlusion)..."
            "$PYTHON_BIN" "$SCRIPT_DIR/run_presentation_demo.py" --mode replay --occlusion 0.50
            read -rp "Press Enter to return to menu..."
            ;;
        2)
            echo ""
            echo "[*] Launching Live Webcam Demo (Camera 0)..."
            echo "[*] Controls: Press 'T' for surgical tool box, 'O' to cycle occlusion, 'Q' to quit."
            "$PYTHON_BIN" "$SCRIPT_DIR/run_presentation_demo.py" --mode webcam --camera 0
            read -rp "Press Enter to return to menu..."
            ;;
        3)
            echo ""
            echo "[*] Launching Mode B UDP Telemetry Receiver (Port 5005)..."
            "$PYTHON_BIN" "$SCRIPT_DIR/run_presentation_demo.py" --mode udp --udp-port 5005 --occlusion 0.50
            read -rp "Press Enter to return to menu..."
            ;;
        4)
            echo ""
            echo "[*] Launching Extreme 70% Occlusion Stress Test..."
            "$PYTHON_BIN" "$SCRIPT_DIR/run_presentation_demo.py" --mode replay --occlusion 0.70
            read -rp "Press Enter to return to menu..."
            ;;
        5)
            echo ""
            echo "[*] Running Kinematics Unit Tests..."
            "$PYTHON_BIN" "$SCRIPT_DIR/tests/test_kinematics.py"
            echo ""
            echo "[*] Running Mode B UDP Diagnostics..."
            "$PYTHON_BIN" "$SCRIPT_DIR/data/iphone_parser.py" --mode test
            read -rp "Press Enter to return to menu..."
            ;;
        6)
            echo ""
            echo "[*] Opening Presentation Slides..."
            if command -v open &>/dev/null; then
                open "$SCRIPT_DIR/Final_Presentation_Slides.pdf"
            elif command -v xdg-open &>/dev/null; then
                xdg-open "$SCRIPT_DIR/Final_Presentation_Slides.pdf"
            fi
            ;;
        7)
            echo ""
            echo "[*] Opening IEEE Project Report..."
            if command -v open &>/dev/null; then
                open "$SCRIPT_DIR/Final_Project_Report.pdf"
            elif command -v xdg-open &>/dev/null; then
                xdg-open "$SCRIPT_DIR/Final_Project_Report.pdf"
            fi
            ;;
        8)
            echo ""
            echo "[*] Opening Examiner 1-Page Cheat Sheet..."
            if command -v open &>/dev/null; then
                open "$SCRIPT_DIR/docs/EXAMINER_CHEATSHEET.md"
            elif command -v xdg-open &>/dev/null; then
                xdg-open "$SCRIPT_DIR/docs/EXAMINER_CHEATSHEET.md"
            fi
            ;;
        9)
            echo ""
            echo "[*] Opening Slide-by-Slide Speaker Script..."
            if command -v open &>/dev/null; then
                open "$SCRIPT_DIR/docs/Slide_By_Slide_Presentation_Guide.md"
            elif command -v xdg-open &>/dev/null; then
                xdg-open "$SCRIPT_DIR/docs/Slide_By_Slide_Presentation_Guide.md"
            fi
            ;;
        [qQ])
            echo "Exiting launcher. Good luck with the presentation!"
            exit 0
            ;;
        *)
            echo "Invalid selection."
            sleep 1
            ;;
    esac
done
