#!/bin/bash

echo "======================================================================="
echo "IAM VALIDATION TEST SUITE"
echo "======================================================================="
echo ""
echo "Running comprehensive validation tests..."
echo "Estimated time: 1-2 minutes"
echo ""

# Record start time
start_time=$(date +%s)

# Run main validation test
echo ">>> Running main validation test (test_03_final.py)"
echo ""
python tests/test_03_final.py

# Check if it succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "✅ ALL TESTS COMPLETED SUCCESSFULLY"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "❌ TESTS FAILED - See errors above"
    echo "======================================================================="
    exit 1
fi

# Record end time
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo ""
echo "Total runtime: ${elapsed} seconds"
echo ""
echo "Results saved to: results/validation_results.npz"
echo ""
echo "For detailed analysis, see: tests/test_03_final.py"
echo "For reproducibility guide, see: IAM_Reproducibility_Quick_Guide.pdf"
echo ""
echo "======================================================================="
