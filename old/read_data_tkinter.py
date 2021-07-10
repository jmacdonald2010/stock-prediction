from database_functions import create_db_connection
from database_functions import execute_query
from database_functions import read_query
import time
from getpass import getpass
import config
import tkinter as tk

main = tk.Tk()

greeting = tk.Label(text="Yeet")
greeting.pack()
main.title("Yeet")
yeet_button = tk.Button(main, text="Yeet", width=25, command=main.destroy)
yeet_button.pack()

main.mainloop()