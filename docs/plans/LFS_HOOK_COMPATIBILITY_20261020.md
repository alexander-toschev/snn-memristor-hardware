# Стабильные Git LFS hooks при разных версиях

На Raspberry Git LFS3.6.1 автоматически заменил четыре tracked hook шаблона,
созданных Git LFS2.13.3 на Windows. `filter-process` вызывает установку hooks;
известный старый шаблон считается обновляемым. Поэтому даже обычный `git diff`
менял tracked файлы и строгий publication preflight правильно останавливался.

Подтверждение в официальных исходниках:
[3.6.1 hook matching/upgrades](https://github.com/git-lfs/git-lfs/blob/v3.6.1/lfs/hook.go),
[3.6.1 filter-process](https://github.com/git-lfs/git-lfs/blob/v3.6.1/commands/command_filter_process.go),
[2.13.3 hook matching](https://github.com/git-lfs/git-lfs/blob/v2.13.3/lfs/hook.go).
Это проблема принадлежности шаблона, не причина ослаблять SHA или dirty checks.

Четыре hooks теперь принадлежат репозиторию: явный комментарий, обязательная
проверка наличия `git-lfs`, затем `exec git lfs <hook> "$@"`.
Это сохраняет аргументы, stdin pre-push и исходный exit status. Отсутствие
Git LFS возвращает2. Custom bytes не совпадают с обновляемыми шаблонами обеих
версий; всё необходимое LFS-действие остаётся. Pre-commit/Gitleaks не меняются.
Publication wrapper по-прежнему требует точные SHA и полный список пяти hooks,
а на POSIX также executable permission. Неизвестные hooks или чужие tracked
изменения блокируют запуск/публикацию.

`git lfs update/install` может отказать с сообщением о custom hook — это ожидаемый
отказ перезаписать repo-owned файл. Не применять `--force`, не удалять hooks,
не игнорировать результат такой команды и не восстанавливать bytes после каждого
Git-вызова. В обычном filter-process собственный installer не заменяет custom
hooks; успешность самого фильтра и Git-команд проверяется отдельно.

Offline-тесты `tests/test_lfs_owned_hooks.py` проверяют реальный локальный
Git LFS2.13.3 filter-process, pointer, commit, три `git diff` и неизменность
hooks; explicit update должен отказать, сохранив bytes. Отдельные shell stubs
проверяют точные argv/stdin, propagation exit0/23 и missing-LFS exit2.
Это не remote upload и не аппаратный эксперимент.

Перед аппаратным запуском основной оператор обязан на Pi с фактическим3.6.1
повторить обычный `git diff`, строгий `git_preflight` и `verify_hooks`,
проверив отсутствие tracked изменений. Локальный тест2.13.3 сам по себе не
является свидетельством выполнения3.6.1 на Pi. Frozen acquisition не меняется.
