import dill

import sys
fpy = open(sys.argv[1], 'r')


fpick_stand =  open(sys.argv[2] + '.standalone.dill', 'wb')
fpick_stack =  open(sys.argv[2] + '.stack.dill', 'wb')
fpick_comp  =  open(sys.argv[2] + '.comp.dill', 'wb')



exec(fpy.read())




dill.dump(standalone, fpick_stand)
dill.dump(stack, fpick_stack)
dill.dump(compare, fpick_comp)

fpick_stand.close()
fpick_stack.close()
fpick_comp .close()