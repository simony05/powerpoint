import pyautogui

screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
print(screenWidth)
print(screenHeight)

pyautogui.moveTo(100, 150)
pyautogui.click()
pyautogui.press('right') 
pyautogui.moveTo(600, 700)
pyautogui.click()
pyautogui.write('a')
pyautogui.press('enter') 