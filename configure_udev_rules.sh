#!/bin/bash


function set_udev_rules() {

    rules_path=/etc/udev/rules.d/arducam-flash.rules

    rules_content='SUBSYSTEM=="usb",ENV{DEVTYPE}=="usb_device",ATTRS{idVendor}=="0c40",MODE="0666"'

    echo -e $rules_content | sudo tee $rules_path
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo "udev rules set successfully!"
}

set_udev_rules
