jq -s '.' $(find ../*/*/*/*/results.jsonl) > all_results.json
dasel -r json -w csv < all_results.json > all_results.csv
