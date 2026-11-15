# Jetson Orin Nano USB & Hardware Bring-Up Brief

**Target Machine:** Linux Mint PC running Antigravity CLI (`agy`)  
**Hardware Connected:** 
1. 32GB USB flash drive plugged into Linux Mint USB port (target boot drive for Jetson).
2. Jetson Orin Nano Developer Kit connected to Linux Mint via USB-C to USB-A cable.

---

## 1. Context & Diagnosis
- The Jetson Orin Nano previously hung on boot with message:
  ```text
  L4TLauncher: Attempting Recovery Boot
  EFI stub: Booting Linux Kernel...
  ...
  EFI stub: Exiting boot services...
  ```
  followed by a blank screen with a blinking underscore (`_`).
- **Root Causes:**
  1. **UEFI Boot-Loop Lockout:** UEFI marks `OS chain A status` as `Unbootable` after 3 consecutive boot failures (which occurred when the system looked for the missing MicroSD card `mmcblk0p1`).
  2. **Device Specifier Mismatch:** Early UEFI/kernel boot requires partition unique IDs (`root=PARTUUID=<uuid>`), not device nodes like `root=/dev/sda1`.
  3. **First-Boot Lockout:** NVIDIA `nv-oem-config` GUI wizard hangs when headless or without a supported DisplayPort monitor, masking standard virtual terminals (`tty2`).

---

## 2. Automated Tasks for Linux Mint Antigravity Agent

When Antigravity runs on Linux Mint, execute these steps:

### Task A: Fix the 32GB USB Drive
1. Locate the USB root partition labeled `APP` (or the large ext4 partition, e.g. `/dev/sdb1` or `/dev/sdc1`).
2. Run `blkid -s PARTUUID -o value <partition>` to get the exact partition UUID (`PARTUUID`).
3. Mount the partition to `/mnt/jetson`.
4. Edit `/mnt/jetson/boot/extlinux/extlinux.conf`:
   - Replace any `root=/dev/[^ ]*` with `root=PARTUUID=<PARTUUID>`.
   - Ensure the APPEND line has:
     `root=PARTUUID=<PARTUUID> rw rootwait rootdelay=5 console=ttyTCU0,115200 console=tty0 fbcon=map:0 net.ifnames=0`
   - Remove `quiet`.
5. Edit `/mnt/jetson/boot/grub/grub.cfg` and any EFI `grub.cfg` to ensure all `root=/dev/...` references use `root=PARTUUID=<PARTUUID>`.
6. Pre-configure user `jetson` (password: `jetson`):
   - Ensure `jetson` exists in `/mnt/jetson/etc/passwd` and `/mnt/jetson/etc/shadow`.
   - Add to groups: `sudo`, `adm`, `video`, `audio`, `dialout`.
   - Add `/mnt/jetson/etc/sudoers.d/99-jetson`: `jetson ALL=(ALL) NOPASSWD:ALL`.
7. Disable blocking first-boot services:
   - Symlink `/dev/null` to:
     - `nv-oem-config.service`
     - `nv-oem-config-gui.service`
     - `nvresizefs.service`
     - `nv-l4t-bootloader-config.service`
8. Enable auto-login on `getty@tty1` and ensure `ssh.service` is enabled.
9. Sync and unmount `/mnt/jetson`.

### Task B: Serial Console Verification
1. Verify the Jetson USB-C serial port is visible at `/dev/ttyACM0`.
2. Ensure `screen` is installed (`sudo apt-get install -y screen`).
3. Instruct user to plug the 32GB USB drive into the Jetson's blue USB 3.2 port and run:
   ```bash
   sudo screen /dev/ttyACM0 115200
   ```
4. Instruct user to cycle power on the Jetson, press `ESC` when prompted, and in the UEFI BIOS menu:
   - Go to `Device Manager` -> `NVIDIA Configuration` -> `L4T Configuration`.
   - Set `OS chain A status` from `Unbootable` to `Normal`.
   - Set `Reset OS chain A counter` to `Reset` (or `0`).
   - Press `F10` to save, `ESC` to exit, and boot!
