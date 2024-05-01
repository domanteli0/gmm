
def seconds_to_time(seconds):
  s = int(seconds) % 60
  m = int(seconds) // 60
  if m < 1:
    return f'{s}s'
  h = m // 60
  m = m % 60
  if h < 1:
    return f'{m}m{s}s'
  return f'{h}h{m}m{s}s'