#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
FEATURE_COLS="PM25,PM10,Temperature,Relative Humidity,Pressure,hour,month"

ALL_MODELS=(
    lstm
    stacked_lstm
    gru
    xgboost
    peak_classifier
    hybrid_ensemble
    cnn_recurrent_attention
    two_stage
    expanding_window_cv
    tuned_lstm
    tuned_stacked_lstm
    tuned_gru
    tuned_xgboost
)

read_horizon() {
    local answer
    while true; do
        if ! read -r -p "Forecast horizon [1 or 6]: " answer; then
            printf '\nInput ended before a horizon was selected.\n' >&2
            exit 1
        fi
        if [[ "$answer" == "1" || "$answer" == "6" ]]; then
            HORIZON="$answer"
            return
        fi
        printf 'Please enter 1 or 6.\n' >&2
    done
}

read_action() {
    local answer
    while true; do
        if ! read -r -p "Run [model/test/everything]: " answer; then
            printf '\nInput ended before a run type was selected.\n' >&2
            exit 1
        fi
        answer="$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')"
        case "$answer" in
            model|test|everything)
                ACTION="$answer"
                return
                ;;
            *)
                printf 'Please enter model, test, or everything.\n' >&2
                ;;
        esac
    done
}

read_model() {
    local answer
    local compact_answer
    local selection
    local model_index
    local model_number
    local -a selections
    local -A added_models=()

    printf '\nAvailable models:\n'
    for ((model_index = 0; model_index < ${#ALL_MODELS[@]}; model_index++)); do
        printf '  %2d) %s\n' "$((model_index + 1))" "${ALL_MODELS[$model_index]}"
    done
    printf '  all) All models\n'

    while true; do
        if ! read -r -p "Model number(s), comma-separated, or all: " answer; then
            printf '\nInput ended before a model was selected.\n' >&2
            exit 1
        fi
        compact_answer="$(printf '%s' "$answer" | tr -d '[:space:]')"
        compact_answer="$(printf '%s' "$compact_answer" | tr '[:upper:]' '[:lower:]')"
        if [[ "$compact_answer" == "all" ]]; then
            SELECTED_MODELS=("${ALL_MODELS[@]}")
            return
        fi

        IFS=',' read -r -a selections <<< "$compact_answer"
        if ((${#selections[@]} == 0)); then
            printf 'Enter one or more model numbers, separated by commas, or all.\n' >&2
            continue
        fi

        SELECTED_MODELS=()
        added_models=()
        for selection in "${selections[@]}"; do
            if [[ ! "$selection" =~ ^[0-9]+$ ]]; then
                printf 'Invalid selection "%s". Enter model numbers separated by commas, or all.\n' "$selection" >&2
                SELECTED_MODELS=()
                break
            fi
            model_number=$((10#$selection))
            if ((model_number < 1 || model_number > ${#ALL_MODELS[@]})); then
                printf 'Model number must be between 1 and %s.\n' "${#ALL_MODELS[@]}" >&2
                SELECTED_MODELS=()
                break
            fi

            selection="${ALL_MODELS[$((model_number - 1))]}"
            if [[ -z "${added_models[$selection]+x}" ]]; then
                SELECTED_MODELS+=("$selection")
                added_models[$selection]=1
            fi
        done

        if ((${#SELECTED_MODELS[@]} > 0)); then
            return
        fi
    done
}

run_model() {
    local model="$1"
    local results_dir="results/t${HORIZON}/${model}"
    local script

    case "$model" in
        lstm) script="main program/main.py" ;;
        stacked_lstm) script="main program/stacked_main.py" ;;
        gru) script="main program/run_gru.py" ;;
        xgboost) script="main program/run_xgb.py" ;;
        peak_classifier) script="main program/run_xgb_peak_classifier.py" ;;
        hybrid_ensemble) script="main program/run_hybrid_ensemble.py" ;;
        cnn_recurrent_attention) script="main program/run_cnn_recurrent_attention.py" ;;
        two_stage) script="main program/run_walk_forward_two_stage.py" ;;
        expanding_window_cv) script="main program/run_expanding_window_cv.py" ;;
        tuned_lstm) script="main program/tune_main.py" ;;
        tuned_stacked_lstm) script="main program/tune_stacked_main.py" ;;
        tuned_gru) script="main program/tune_gru_main.py" ;;
        tuned_xgboost) script="main program/tune_xgb_main.py" ;;
        *)
            printf 'Unknown model: %s\n' "$model" >&2
            return 2
            ;;
    esac

    printf '\n=== Running %s for t+%s ===\n' "$model" "$HORIZON"
    HORIZON="$HORIZON" \
        FEATURE_COLS="$FEATURE_COLS" \
        RESULTS_DIR="$results_dir" \
        "$PYTHON_BIN" "$script"
}

run_tests() {
    printf '\n=== Running test suite for t+%s ===\n' "$HORIZON"
    HORIZON="$HORIZON" \
        FEATURE_COLS="$FEATURE_COLS" \
        RESULTS_DIR="results/t${HORIZON}/tests" \
        "$PYTHON_BIN" -m unittest discover -s tests -v
}

read_horizon
read_action

case "$ACTION" in
    model)
        read_model
        for model in "${SELECTED_MODELS[@]}"; do
            run_model "$model"
        done
        ;;
    test)
        run_tests
        ;;
    everything)
        SELECTED_MODELS=("${ALL_MODELS[@]}")
        for model in "${SELECTED_MODELS[@]}"; do
            run_model "$model"
        done
        run_tests
        ;;
esac

printf '\nAll requested commands completed successfully.\n'
