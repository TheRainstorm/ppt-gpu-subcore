from simian import MPI

defaultMpichLibName = "/usr/lib/x86_64-linux-gnu/libmpich.so"
mpi = MPI(defaultMpichLibName)

print(mpi.rank(), mpi.size())