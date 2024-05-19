def init():
  classes_no_background = [
    'Apple',
    'Axe',
    'Cat',
    'Teapot',
    'Spoon',
  ]

  classes = classes_no_background.copy()
  classes.insert(0, "Background")

  classes_dict = {label: ix for ix, label in enumerate(classes)}

  num_classes = len(classes)
  return classes_no_background, classes, classes_dict, num_classes

classes_no_background, classes, classes_dict, num_classes = init()
