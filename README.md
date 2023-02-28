Хотим сделать бота, который отвечает на часто задаваемые вопросы. Для этого у нас
есть какой-то чёрный ящик, который для данной реплики возвращает наиболее
подходящий ответ и score, насколько по мнению этого чёрного ящика этот ответ
подходит (от 0 до 100). При этом мы понимаем, что бот не сможет отвечать на все
вопросы, и поэтому делаем так:
1) Если бот сильно уверен в ответе (score высокий), то даём пользователю ответ
2) Если бот совсем не уверен в ответе (score низкий), то переводим пользователя
на оператора
3) Если score средний, то просим пользователя переформулировать вопрос.
Нужно выяснить, какой score считать низким, какой средним, а какой высоким. Мы
провели эксперимент: попросили живых людей задать вопрос боту, а потом указать,
что было бы правильно после такого вопроса сделать: ответить, переспросить или
перевести на оператора. Мы записали в таблицу score от чёрного ящика и
предложенное тестером действие. (table.csv)
Ваша задача - реализовать алгоритм, будет принимать на вход score и возвращать
следующее действие: вывести ответ пользователю, переспросить пользователя или
перевести на оператора. Обоснуйте, почему был выбран именно такой алгоритм и
именно такие параметры для него.
Предложите метрику для определения точности своего алгоритма и посчитайте её
значение для данных из table.csv