def init():
  classes_no_background = [
    'Apple',
    'Cucumber',
    'Mushroom',
    'Pumpkin',
    'Watermelon',
  ]

  classes = classes_no_background.copy()
  classes.insert(0, "Background")

  classes_dict = {label: ix for ix, label in enumerate(classes)}

  num_classes = len(classes)
  return classes_no_background, classes, classes_dict, num_classes

classes_no_background, classes, classes_dict, num_classes = init()
