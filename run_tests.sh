#!/bin/bash

# Set up color codes for better visibility
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting test suite for Feel.io Emotion Recognition Project${NC}\n"

# Create a timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="test_results_${TIMESTAMP}.log"

# Function to run a test file and check its status
run_test() {
    local test_file=$1
    echo -e "\n${YELLOW}Running tests from: ${test_file}${NC}"
    python -m pytest $test_file -v >> $LOG_FILE 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Tests in $test_file passed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Tests in $test_file failed${NC}"
        echo -e "${YELLOW}Check $LOG_FILE for details${NC}"
        return 1
    }
}

# Initialize error counter
ERRORS=0

# Create directory for test logs if it doesn't exist
mkdir -p test_logs

echo "Test execution started at $(date)" > test_logs/$LOG_FILE
echo "----------------------------------------" >> test_logs/$LOG_FILE

# Run each test file
echo -e "\n${YELLOW}Running all tests...${NC}\n"

# List of test files
TEST_FILES=(
    "tests/test_dataset.py"
    "tests/test_model.py"
    "tests/test_training.py"
)

# Run tests and collect results
for test_file in "${TEST_FILES[@]}"; do
    if [ -f "$test_file" ]; then
        run_test $test_file
        ERRORS=$((ERRORS + $?))
    else
        echo -e "${RED}Error: Test file $test_file not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Print summary
echo -e "\n${YELLOW}Test Summary:${NC}"
echo "----------------------------------------"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All tests passed successfully!${NC}"
else
    echo -e "${RED}$ERRORS test suite(s) reported errors${NC}"
    echo -e "${YELLOW}Check test_logs/$LOG_FILE for detailed information${NC}"
fi

# Exit with appropriate status code
exit $ERRORS