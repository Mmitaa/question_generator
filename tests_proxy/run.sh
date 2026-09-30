#!/bin/bash
# Прогоняет все тесты прокси и печатает общий итог.
cd "$(dirname "$0")" || exit 1
ok=0; bad=0
for test in test_config.py test_cluster.py test_training.py test_admin.py test_http.py; do
  line=$(python3 "$test" 2>&1 | grep -o 'итого: [0-9]* ок, [0-9]* провалов' | tail -1)
  printf '%-20s %s\n' "$test" "${line:-НЕ ОТРАБОТАЛ}"
  ok=$((ok + $(echo "$line" | grep -o '^итого: [0-9]*' | grep -o '[0-9]*' || echo 0)))
  bad=$((bad + $(echo "$line" | grep -o '[0-9]* провалов' | grep -o '[0-9]*' || echo 1)))
done
echo "──────────────────────────────────"
echo "ВСЕГО: $ok ок, $bad провалов"
[ "$bad" -eq 0 ]
