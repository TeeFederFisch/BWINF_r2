##################################################################################################################################
------------------------------------------------------config.txt Information------------------------------------------------------
##################################################################################################################################

Die erste Zeile gibt an ob die berechneten Ergebnisse mit Python visualisiert werden.
Auch können dann mit Hilfe des pyturtle Fensters eigene Beispiele eingegeben werden.

In der zweiten Zeile wird der Modus festgelegt mit dem die Beispieldateien bearbeitet werden.
Optionen: normal, expensive, cached, optimal

In der dritten Zeile wird der Modus festgelegt mit dem die eigenen Beispiele bearbeitet werde. (Nur mit python=true möglich)
Optionen: normal, expensive, cached, optimal

Jede weitere Zeile wird als Dateipfad zu einer Inputdatei ausgewertet. (Error bei ungültigem Pfad)
Eine selber erstellte Inputdatei muss die gleiche Struktur wie die vom Wettbewerb vorgegebenen Beispiele haben.


##################################################################################################################################
------------------------------------------------------------Bedienung-------------------------------------------------------------
##################################################################################################################################

Für jede der in config.txt spezifizierten Inputdateien wird eine Outputdatei <dateipfad>_res.txt erstellt.
Diese Datei hat die selbe Struktur wie die Inputdatei, allerdings sind die Punkte in der Reihenfolge enthalten,
in der man ihnen folgen muss, um den Lösungspfad zu erhalten.

Um die pyturtle GUI zu verwenden, muss in config.txt #python=true gesetzt sein.
Die Lösungen zu den Dateien werden dann grafisch dargestellt. Weitere Informationen auf der Konsole ausgegeben. 
Erst wenn hier "Finished <dateipfad> [...]" zu sehen ist, kann die Berechnung der nächsten Datei per Tastendruck gestartet werden.

Wenn "Try drawing something yourself now!" auf der Konsole ausgegeben wird, kann man mit der linken Maustaste Punkte auf das
pyturtle Fenster malen. Auf Tastatureingabe wird der in config.txt #custom_input_mode angegebene Algorithmus auf die
gezeichnete Wolke angewandt und die Lösung genau wie bei den Dateien angezeigt. Hier wird allerdings keine Lösungsdatei erstellt.
Mit einem erneuten Tastendruck kann das Fenster nach der Berechnung wieder geleert werden. 
Um das Fenster zu schließen zwei Mal eine Taste drücken.