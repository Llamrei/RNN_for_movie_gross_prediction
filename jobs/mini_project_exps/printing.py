# Modified print row function that wraps the output
# Replaces layer_utils printing stuff
positions = [30, 60, 90]
print('_'*positions[-1])
def print_row(fields, positions):
    left_to_print = [str(x) for x in fields]
    while any(left_to_print):
      line = ''
      for i in range(len(left_to_print)):
        if i > 0:
          start_pos = positions[i-1]
        else:
          start_pos = 0
        end_pos = positions[i]
        # Leave room for a space
        delta = end_pos - start_pos - 1
        fit_into_line = left_to_print[i][:delta]
        line += fit_into_line
        line += ' '
        left_to_print[i] = left_to_print[i][delta:]

        # Pad out ot the next position
        line += ' ' * (positions[i] - len(line))
        # line = line[:-1]+'|'
      print(line)
a = 'something super long that needs to be wrapped all nicely please i beg you just work you trash'
print_row([a,a,a], positions)