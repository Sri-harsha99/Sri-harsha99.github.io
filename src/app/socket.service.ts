import { Injectable, ElementRef } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { Observable } from 'rxjs';
import { HttpClient, HttpParams } from "@angular/common/http";
import { SocketIoConfig, SocketIoModule } from 'ngx-socket-io';
import { SignalingService } from './signaling.service';

@Injectable({
  providedIn: 'root'
})
export class SocketService {
  stream;
  private socket: Socket;
  connection: RTCPeerConnection;
  configuration: RTCConfiguration = {
    iceServers: [
      {
        urls: [
          'stun:stun1.l.google.com:19302',
          'stun:stun2.l.google.com:19302',
        ],
      },
    ],
    iceCandidatePoolSize: 10,
  };
  messages = [];
  onlineUsers = [];
  nodeURL = 'http://localhost:3000/';
  constructor(private http: HttpClient,
    private signalingService: SignalingService) {
    this.socket = io('http://localhost:3000');

    this.socket.on('chatMessage', (data: string) => {
      this.messages.push(data);
    });

    this.socket.on('userList', (data) => {
      console.log(data);
      this.onlineUsers.push(data);
    });
  }

  // Inside SocketService class

  getUsers(): Observable<string[]> {
    return new Observable<string[]>(observer => {
      this.socket.on('userList', users => {
        observer.next(users);
      });
    });
  }


  test(){
    console.log('test');
    return this.http.get<any>(this.nodeURL+'/test');
  }

  joinUser(displayName: string) {
    this.socket.emit('join', {displayName});
  }

  sendPrivateMessage(message: string, displayName: string, recipient: string) {
    this.socket.emit('privateMessage', { message, displayName, recipient });
  }


  sendMessage(message: string, displayName: string, recipient?: string) {
    if(recipient) {
      this.sendPrivateMessage(message, displayName, recipient);
      return;
    }
    this.socket.emit('chatMessage', { message, displayName });
  }

  getMessage(): Observable<any> {
    return new Observable<any>(observer => {
      this.socket.on('chatMessage', data => {
        observer.next(data);
      });
    });
  }
  private async _initConnection(remoteVideo: ElementRef): Promise<void> {
    this.connection = new RTCPeerConnection(this.configuration);

    await this._getStreams(remoteVideo);

    this._registerConnectionListeners();
  }

  public async makeCall(remoteVideo: ElementRef): Promise<void> {
    await this._initConnection(remoteVideo);

    const offer = await this.connection.createOffer();

    await this.connection.setLocalDescription(offer);

    this.signalingService.sendMessage({ type: 'offer', offer });
  }

  public async handleOffer(
    offer: RTCSessionDescription,
    remoteVideo: ElementRef
  ): Promise<void> {
    await this._initConnection(remoteVideo);

    await this.connection.setRemoteDescription(
      new RTCSessionDescription(offer)
    );

    const answer = await this.connection.createAnswer();

    await this.connection.setLocalDescription(answer);

    this.signalingService.sendMessage({ type: 'answer', answer });
  }

  public async handleAnswer(answer: RTCSessionDescription): Promise<void> {
    await this.connection.setRemoteDescription(
      new RTCSessionDescription(answer)
    );
  }

  public async handleCandidate(candidate: RTCIceCandidate): Promise<void> {
    if (candidate) {
      await this.connection.addIceCandidate(new RTCIceCandidate(candidate));
    }
  }

  private _registerConnectionListeners(): void {
    this.connection.onicegatheringstatechange = (ev: Event) => {
      console.log(
        `ICE gathering state changed: ${this.connection.iceGatheringState}`
      );
    };

    this.connection.onconnectionstatechange = () => {
      console.log(
        `Connection state change: ${this.connection.connectionState}`
      );
    };

    this.connection.onsignalingstatechange = () => {
      console.log(`Signaling state change: ${this.connection.signalingState}`);
    };

    this.connection.oniceconnectionstatechange = () => {
      console.log(
        `ICE connection state change: ${this.connection.iceConnectionState}`
      );
    };
    this.connection.onicecandidate = (event) => {
      if (event.candidate) {
        const payload = {
          type: 'candidate',
          candidate: event.candidate.toJSON(),
        };
        this.signalingService.sendMessage(payload);
      }
    };
  }

  private async _getStreams(remoteVideo: ElementRef): Promise<void> {
    
    try{
      if (this.stream){
        /* On some android devices, it is necessary to stop the previous track*/
        this.stream.getTracks().forEach(t => t.stop());
      }
      this.stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true,
    });
    } catch(e){
      console.log(e);
    }

    const remoteStream = new MediaStream();

    remoteVideo.nativeElement.srcObject = remoteStream;

    this.connection.ontrack = (event) => {
      event.streams[0].getTracks().forEach((track) => {
        remoteStream.addTrack(track);
      });
    };

    this.stream.getTracks().forEach((track) => {
      this.connection.addTrack(track, this.stream);
    });
  }
}
