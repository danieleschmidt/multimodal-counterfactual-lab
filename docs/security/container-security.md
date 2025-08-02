# Container Security Best Practices

## Overview

Container security is critical for AI/ML applications that handle sensitive data and models. This guide outlines comprehensive security measures implemented for the counterfactual lab container infrastructure.

## Dockerfile Security Enhancements

### Multi-Stage Build Security

```dockerfile
# Secure multi-stage build pattern
FROM python:3.11-slim as builder
LABEL security.scan="enabled"

# Security-focused system updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
```

### Non-Root User Implementation

```dockerfile
# Create restricted user with specific UID
RUN useradd --create-home --shell /bin/bash --uid 10001 app \
    && mkdir -p /home/app/.cache \
    && chown -R app:app /home/app

USER app
WORKDIR /home/app
```

### Security Environment Variables

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
```

## Container Scanning

### Hadolint Integration

```bash
# Install Hadolint for Dockerfile linting
wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64
chmod +x /usr/local/bin/hadolint

# Scan Dockerfile for security issues
hadolint Dockerfile --config .hadolint.yaml
```

### Trivy Security Scanning

```bash
# Install Trivy vulnerability scanner
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan container image for vulnerabilities
trivy image multimodal-counterfactual-lab:latest

# Generate detailed report
trivy image --format json --output security-report.json multimodal-counterfactual-lab:latest
```

### Snyk Container Scanning

```bash
# Install Snyk CLI
npm install -g snyk

# Authenticate with Snyk
snyk auth

# Scan container for vulnerabilities
snyk container test multimodal-counterfactual-lab:latest

# Monitor container in production
snyk container monitor multimodal-counterfactual-lab:latest
```

## Base Image Security

### Distroless Images (Recommended)

```dockerfile
# Use distroless for production (ultra-minimal attack surface)
FROM gcr.io/distroless/python3-debian11:latest as production-distroless

# Copy only necessary files
COPY --from=builder --chown=nonroot:nonroot /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=nonroot:nonroot /app/src ./src

USER nonroot
EXPOSE 8501
```

### Alpine Linux Alternative

```dockerfile
FROM python:3.11-alpine as alpine-builder

# Security updates for Alpine
RUN apk update && apk upgrade \
    && apk add --no-cache \
        build-base \
        linux-headers \
        ca-certificates \
    && rm -rf /var/cache/apk/*
```

## Runtime Security

### Container Isolation

```bash
# Run with security constraints
docker run \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/run \
  --tmpfs /var/lock \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --no-new-privileges \
  --user 10001:10001 \
  --security-opt no-new-privileges:true \
  --security-opt seccomp=seccomp-profile.json \
  multimodal-counterfactual-lab:latest
```

### AppArmor Profile

```bash
# Create AppArmor profile for container
cat > /etc/apparmor.d/docker-counterfactual-lab << 'EOF'
#include <tunables/global>

profile docker-counterfactual-lab flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_ptrace,
  deny capability sys_module,
  
  # Allow necessary file access
  /home/app/** rw,
  /usr/local/lib/python3.11/** r,
  /tmp/** rw,
  
  # Deny sensitive paths
  deny /proc/sys/** w,
  deny /sys/** w,
}
EOF

# Load profile
apparmor_parser -r /etc/apparmor.d/docker-counterfactual-lab
```

### Seccomp Profile

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat", "lstat",
        "poll", "lseek", "mmap", "mprotect", "munmap", "brk",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "ioctl",
        "pread64", "pwrite64", "readv", "writev", "access", "pipe",
        "select", "sched_yield", "mremap", "msync", "mincore",
        "madvise", "shmget", "shmat", "shmctl", "dup", "dup2",
        "pause", "nanosleep", "getitimer", "alarm", "setitimer",
        "getpid", "sendfile", "socket", "connect", "accept", "sendto",
        "recvfrom", "sendmsg", "recvmsg", "shutdown", "bind", "listen",
        "getsockname", "getpeername", "socketpair", "setsockopt",
        "getsockopt", "clone", "fork", "vfork", "execve", "exit",
        "wait4", "kill", "uname", "semget", "semop", "semctl",
        "shmdt", "msgget", "msgsnd", "msgrcv", "msgctl", "fcntl",
        "flock", "fsync", "fdatasync", "truncate", "ftruncate",
        "getdents", "getcwd", "chdir", "fchdir", "rename", "mkdir",
        "rmdir", "creat", "link", "unlink", "symlink", "readlink",
        "chmod", "fchmod", "chown", "fchown", "lchown", "umask",
        "gettimeofday", "getrlimit", "getrusage", "sysinfo", "times",
        "ptrace", "getuid", "syslog", "getgid", "setuid", "setgid",
        "geteuid", "getegid", "setpgid", "getppid", "getpgrp",
        "setsid", "setreuid", "setregid", "getgroups", "setgroups",
        "setresuid", "getresuid", "setresgid", "getresgid", "getpgid",
        "setfsuid", "setfsgid", "getsid", "capget", "capset",
        "rt_sigpending", "rt_sigtimedwait", "rt_sigqueueinfo",
        "rt_sigsuspend", "sigaltstack", "utime", "mknod", "uselib",
        "personality", "ustat", "statfs", "fstatfs", "sysfs",
        "getpriority", "setpriority", "sched_setparam", "sched_getparam",
        "sched_setscheduler", "sched_getscheduler", "sched_get_priority_max",
        "sched_get_priority_min", "sched_rr_get_interval", "mlock",
        "munlock", "mlockall", "munlockall", "vhangup", "modify_ldt",
        "pivot_root", "prctl", "arch_prctl", "adjtimex", "setrlimit",
        "sync", "acct", "settimeofday", "mount", "umount2", "swapon",
        "swapoff", "reboot", "sethostname", "setdomainname", "iopl",
        "ioperm", "create_module", "init_module", "delete_module",
        "get_kernel_syms", "query_module", "quotactl", "nfsservctl",
        "getpmsg", "putpmsg", "afs_syscall", "tuxcall", "security",
        "gettid", "readahead", "setxattr", "lsetxattr", "fsetxattr",
        "getxattr", "lgetxattr", "fgetxattr", "listxattr", "llistxattr",
        "flistxattr", "removexattr", "lremovexattr", "fremovexattr",
        "tkill", "time", "futex", "sched_setaffinity", "sched_getaffinity",
        "set_thread_area", "io_setup", "io_destroy", "io_getevents",
        "io_submit", "io_cancel", "get_thread_area", "lookup_dcookie",
        "epoll_create", "epoll_ctl_old", "epoll_wait_old", "remap_file_pages",
        "getdents64", "set_tid_address", "restart_syscall", "semtimedop",
        "fadvise64", "timer_create", "timer_settime", "timer_gettime",
        "timer_getoverrun", "timer_delete", "clock_settime", "clock_gettime",
        "clock_getres", "clock_nanosleep", "exit_group", "epoll_wait",
        "epoll_ctl", "tgkill", "utimes", "vserver", "mbind", "set_mempolicy",
        "get_mempolicy", "mq_open", "mq_unlink", "mq_timedsend",
        "mq_timedreceive", "mq_notify", "mq_getsetattr", "kexec_load",
        "waitid", "add_key", "request_key", "keyctl", "ioprio_set",
        "ioprio_get", "inotify_init", "inotify_add_watch", "inotify_rm_watch",
        "migrate_pages", "openat", "mkdirat", "mknodat", "fchownat",
        "futimesat", "newfstatat", "unlinkat", "renameat", "linkat",
        "symlinkat", "readlinkat", "fchmodat", "faccessat", "pselect6",
        "ppoll", "unshare", "set_robust_list", "get_robust_list",
        "splice", "tee", "sync_file_range", "vmsplice", "move_pages",
        "utimensat", "epoll_pwait", "signalfd", "timerfd_create", "eventfd",
        "fallocate", "timerfd_settime", "timerfd_gettime", "accept4",
        "signalfd4", "eventfd2", "epoll_create1", "dup3", "pipe2",
        "inotify_init1", "preadv", "pwritev", "rt_tgsigqueueinfo",
        "perf_event_open", "recvmmsg", "fanotify_init", "fanotify_mark",
        "prlimit64", "name_to_handle_at", "open_by_handle_at", "clock_adjtime",
        "syncfs", "sendmmsg", "setns", "getcpu", "process_vm_readv",
        "process_vm_writev", "kcmp", "finit_module"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
```

## Image Signing and Verification

### Cosign Integration

```bash
# Install Cosign
go install github.com/sigstore/cosign/cmd/cosign@latest

# Generate key pair
cosign generate-key-pair

# Sign container image
cosign sign --key cosign.key multimodal-counterfactual-lab:latest

# Verify signed image
cosign verify --key cosign.pub multimodal-counterfactual-lab:latest
```

### Notary v2 (Future)

```bash
# Sign with notation (Notary v2)
notation sign multimodal-counterfactual-lab:latest

# Verify signature
notation verify multimodal-counterfactual-lab:latest
```

## Secrets Management

### Docker Secrets

```bash
# Create secret
echo "api_key_value" | docker secret create api_key -

# Use secret in container
docker service create \
  --secret api_key \
  --name counterfactual-service \
  multimodal-counterfactual-lab:latest
```

### External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "counterfactual-lab"
```

## Network Security

### Container Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: counterfactual-lab-netpol
spec:
  podSelector:
    matchLabels:
      app: counterfactual-lab
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8501
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
    - protocol: TCP  
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### TLS Configuration

```dockerfile
# Add TLS certificate management
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

## Monitoring and Compliance

### Falco Rules

```yaml
# /etc/falco/falco_rules.local.yaml
- rule: Detect Privilege Escalation in Container
  desc: Detect attempts to escalate privileges in counterfactual lab container
  condition: >
    spawned_process and
    container.image.repository contains "counterfactual-lab" and
    (proc.name in (su, sudo, setuid_binaries) or
     proc.args contains "chmod +s")
  output: >
    Privilege escalation attempt in counterfactual lab container
    (user=%user.name command=%proc.cmdline container=%container.name)
  priority: WARNING

- rule: Unexpected Network Connection
  desc: Detect unexpected outbound connections from container
  condition: >
    outbound and
    container.image.repository contains "counterfactual-lab" and
    not fd.sport in (80, 443, 53)
  output: >
    Unexpected network connection from counterfactual lab
    (connection=%fd.name container=%container.name)
  priority: WARNING
```

### OPA/Gatekeeper Policies

```yaml
apiVersion: templates.gatekeeper.sh/v1beta1
kind: ConstraintTemplate
metadata:
  name: containerexecute
spec:
  crd:
    spec:
      names:
        kind: ContainerExecute
      validation:
        properties:
          allowedImages:
            type: array
            items:
              type: string
  targets:
    - target: admission.k8s.gatekeeper.sh
      rego: |
        package containerexecute
        
        violation[{"msg": msg}] {
          container := input.review.object.spec.containers[_]
          not starts_with(container.image, input.parameters.allowedImages[_])
          msg := sprintf("Container image %v is not from allowed registry", [container.image])
        }
```

## CI/CD Security Integration

### GitHub Actions Security

```yaml
name: Container Security Scan
on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build container
        run: docker build -t test-image .
        
      - name: Run Hadolint
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          config: .hadolint.yaml
          
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'test-image'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
          
      - name: Sign container with Cosign
        uses: sigstore/cosign-installer@v3.1.1
        
      - name: Sign image
        run: |
          cosign sign --yes test-image
        env:
          COSIGN_EXPERIMENTAL: 1
```

## Compliance and Auditing

### CIS Benchmarks

```bash
# Install Docker Bench Security
git clone https://github.com/docker/docker-bench-security.git
cd docker-bench-security
sudo sh docker-bench-security.sh

# Specific tests for our container
docker run --rm -it \
  --label docker_bench_security \
  docker/docker-bench-security:latest
```

### NIST Compliance

```yaml
# security-controls.yaml
controls:
  - id: AC-3
    title: Access Enforcement
    implementation: >
      Container runs as non-root user with UID 10001.
      AppArmor profile restricts system access.
      
  - id: SC-8
    title: Transmission Confidentiality
    implementation: >
      All network communication uses TLS 1.3.
      Internal communication encrypted with mTLS.
      
  - id: SI-10
    title: Information Input Validation  
    implementation: >
      Input validation in application layer.
      Container filesystem is read-only except /tmp.
```

## Best Practices Summary

1. **Minimal Base Images**: Use distroless or Alpine Linux
2. **Non-Root Users**: Always run containers as non-root
3. **Read-Only Filesystems**: Mount root filesystem as read-only
4. **Capability Dropping**: Drop all unnecessary Linux capabilities
5. **Security Scanning**: Integrate vulnerability scanning in CI/CD
6. **Image Signing**: Sign and verify container images
7. **Network Policies**: Implement strict network isolation
8. **Secrets Management**: Use external secret management systems
9. **Runtime Security**: Deploy runtime security monitoring
10. **Compliance**: Follow industry security standards and benchmarks

## References

- [NIST Container Security Guide](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-190.pdf)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Container Security Top 10](https://github.com/OWASP/Container-Security-Top-10)
- [NSA Container Security Hardening Guide](https://media.defense.gov/2021/Aug/03/2002820425/-1/-1/0/CTR_Kubernetes%20Hardening%20Guidance.PDF)