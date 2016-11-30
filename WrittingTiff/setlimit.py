import resource
import subprocess
import os
import pwd

print "getrlimit before:", resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
print "getrlimit:", resource.getrlimit(resource.RLIMIT_NOFILE)
print "subprocess:", subprocess.check_output("whoami; ulimit -n", shell=True)

user = "pnaylor"
pwnam = pwd.getpwnam(user)
os.setgid(pwnam.pw_gid)
os.setuid(pwnam.pw_uid)
print "getrlimit:", resource.getrlimit(resource.RLIMIT_NOFILE)
print "subprocess:", subprocess.check_output("whoami; ulimit -n", shell=True)